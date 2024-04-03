#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
from typing import Callable, List

import hydra
import optuna
from omegaconf import DictConfig

# %%
import torch
import torch.nn as nn

# %%
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import AutoResetWrapper

# %%
from bbrl import get_arguments, get_class
from bbrl.agents import TemporalAgent, Agents
from bbrl.workspace import Workspace
from bbrl.agents.gymnasium import ParallelGymAgent
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_algos.models.exploration_agents import EGreedyActionSelector
from bbrl_algos.models.critics import DiscreteQAgent
from bbrl_algos.models.loggers import Logger


# %%
def compute_critic_loss(
    discount_factor, reward, must_bootstrap, action, q_values, q_target=None
):
    """Compute critic loss
    Args:
        discount_factor (float): The discount factor
        reward (torch.Tensor): a (2 × T × B) tensor containing the rewards
        must_bootstrap (torch.Tensor): a (2 × T × B) tensor containing 0 if the episode is completed at time $t$
        action (torch.LongTensor): a (2 × T) long tensor containing the chosen action
        q_values (torch.Tensor): a (2 × T × B × A) tensor containing Q values
        q_target (torch.Tensor, optional): a (2 × T × B × A) tensor containing target Q values

    Returns:
        torch.Scalar: The loss
    """
    if q_target is None:
        q_target = q_values
    max_q = q_target[1].amax(dim=-1).detach()
    target = reward[1] + discount_factor * max_q * must_bootstrap[1]
    act = action[0].unsqueeze(dim=-1)
    qvals = q_values[0].gather(dim=1, index=act).squeeze(dim=1)
    return nn.MSELoss()(qvals, target)


# %%
def compute_critic_loss_ddqn(
    discount_factor, reward, must_bootstrap, action, q_values, q_target=None
):
    # Compute temporal difference
    target_actions = q_values[1].max(1)[1].unsqueeze(-1).detach()
    target_q = q_target.gather(-1, index=target_actions).squeeze()
    target = reward[1].squeeze() + discount_factor * target_q * must_bootstrap[1]
    act = action[0].unsqueeze(dim=-1)
    qvals = q_values[0].gather(dim=1, index=act).squeeze(dim=1)
    critic_loss = nn.MSELoss()(target, qvals)
    return critic_loss


# %%
def make_wrappers(
    autoreset: bool,
) -> List[Callable[[Env], Env]]:
    return [AutoResetWrapper] if autoreset else []


# %%
def make_env(
    identifier: str,
    autoreset: bool,
    **kwargs,
) -> Env:
    env: Env = gym.make(id=identifier, **kwargs)
    wrappers = make_wrappers(
        autoreset=autoreset,
    )
    for wrapper in wrappers:
        env = wrapper(env)
    return env


# %%
def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


# %%
def create_dqn_agent(cfg_algo, train_env_agent, eval_env_agent):
    state_dim, action_dim = train_env_agent.get_obs_and_actions_sizes()

    critic = DiscreteQAgent(
        state_dim=state_dim,
        hidden_layers=list(cfg_algo.architecture.hidden_sizes),
        action_dim=action_dim,
        seed=cfg_algo.seed.q,
    )
    critic_target = copy.deepcopy(critic)

    explorer = EGreedyActionSelector(
        name="action_selector",
        epsilon=cfg_algo.explorer.epsilon_start,
        epsilon_end=cfg_algo.explorer.epsilon_end,
        epsilon_decay=cfg_algo.explorer.decay,
        seed=cfg_algo.seed.explorer,
    )
    q_agent = TemporalAgent(critic)
    q_agent_target = TemporalAgent(critic_target)

    tr_agent = Agents(train_env_agent, critic, explorer)
    ev_agent = Agents(eval_env_agent, critic)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)

    return train_agent, eval_agent, q_agent, q_agent_target


# %%
# Configure the optimizer over the q agent
def setup_optimizer(optimizer_cfg, q_agent):
    optimizer_args = get_arguments(optimizer_cfg)
    parameters = q_agent.parameters()
    optimizer = get_class(optimizer_cfg)(parameters, **optimizer_args)
    return optimizer


# %%
def run_dqn(cfg, logger, trial=None):
    # 1) Create the environment agent
    train_env_agent = ParallelGymAgent(
        make_env_fn=get_class(cfg.gym_env_train),
        num_envs=cfg.algorithm.n_envs_train,
        make_env_args=get_arguments(cfg.gym_env_train),
        seed=cfg.algorithm.seed.train,
    )
    eval_env_agent = ParallelGymAgent(
        make_env_fn=get_class(cfg.gym_env_eval),
        num_envs=cfg.algorithm.n_envs_eval,
        make_env_args=get_arguments(cfg.gym_env_eval),
        seed=cfg.algorithm.seed.eval,
    )

    # 2) Create the DQN-like Agent
    train_agent, eval_agent, q_agent, q_agent_target = create_dqn_agent(
        cfg.algorithm, train_env_agent, eval_env_agent
    )

    # 3) Create the training workspace
    train_workspace = Workspace()  # Used for training

    # 4) Create the replay buffer
    replay_buffer = ReplayBuffer(max_size=cfg.algorithm.buffer.max_size)

    # 5) Configure the optimizer
    optimizer = setup_optimizer(cfg.optimizer, q_agent)

    # 6) Define the steps counters
    nb_steps = 0
    tmp_steps_target_update = 0
    tmp_steps_eval = 0

    while nb_steps < cfg.algorithm.n_steps:
        # Decay the explorer epsilon
        explorer = train_agent.agent.get_by_name("action_selector")
        assert len(explorer) == 1, "There should be only one explorer"
        explorer[0].decay()

        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps_train - 1,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps_train,
            )

        transition_workspace: Workspace = train_workspace.get_transitions(
            filter_key="env/done"
        )

        # Only get the required number of steps
        steps_diff = cfg.algorithm.n_steps - nb_steps
        if transition_workspace.batch_size() > steps_diff:
            for key in transition_workspace.keys():
                transition_workspace.set_full(
                    key, transition_workspace[key][:, :steps_diff]
                )

        nb_steps += transition_workspace.batch_size()

        # Store the transition in the replay buffer
        replay_buffer.put(transition_workspace)

        for _ in range(cfg.algorithm.optim_n_updates):
            # Sample a batch from the replay buffer
            sampled_trans_ws = replay_buffer.get_shuffled(
                cfg.algorithm.buffer.batch_size
            )

            # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
            q_agent(sampled_trans_ws, t=0, n_steps=2, choose_action=False)

            q_values, terminated, reward, action = sampled_trans_ws[
                "q_values",
                "env/terminated",
                "env/reward",
                "action",
            ]

            with torch.no_grad():
                q_agent_target(sampled_trans_ws, t=0, n_steps=2)

            # Compute the target q values
            q_target = sampled_trans_ws["q_values"]

            # Determines whether values of the critic should be propagated
            # True if the episode reached a time limit or if the task was not terminated.
            must_bootstrap = ~terminated

            # Store the replay buffer size
            logger.add_log("replay_buffer_size", replay_buffer.size(), nb_steps)

            if replay_buffer.size() > cfg.algorithm.buffer.learning_starts:

                # funct = compute_critic_loss
                funct = compute_critic_loss_ddqn

                critic_loss = funct(
                    cfg.algorithm.discount_factor,
                    reward,
                    must_bootstrap,
                    action,
                    q_values,
                    q_target,
                )

                # Store the loss
                logger.add_log("critic_loss", critic_loss, nb_steps)

                optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    q_agent.parameters(), cfg.algorithm.max_grad_norm
                )
                optimizer.step()

        # Update the target network
        if (
            nb_steps - tmp_steps_target_update
            > cfg.algorithm.target_critic_update_interval
        ):
            tmp_steps_target_update = nb_steps
            q_agent_target.agent = copy.deepcopy(q_agent.agent)

        # Evaluate the agent
        if nb_steps - tmp_steps_eval > cfg.algorithm.eval_interval:
            tmp_steps_eval = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace,
                t=0,
                stop_variable="env/done",
                choose_action=True,
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()

            logger.log_reward_losses(rewards, nb_steps)

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    return mean


# %%
def get_trial_value(trial: optuna.Trial, cfg: DictConfig, variable_name: str):
    # code suivant assez moche, certes, piste d’amélioration possible
    suggest_type = cfg["suggest_type"]
    args = cfg.keys() - ["suggest_type"]
    args_str = ", ".join([f"{arg}={cfg[arg]}" for arg in args])
    return eval(f'trial.suggest_{suggest_type}("{variable_name}", {args_str})')


def get_trial_config(trial: optuna.Trial, cfg: DictConfig):
    for variable_name in cfg.keys():
        if type(cfg[variable_name]) != DictConfig:
            continue
        else:
            if "suggest_type" in cfg[variable_name].keys():
                cfg[variable_name] = get_trial_value(
                    trial, cfg[variable_name], variable_name
                )
            else:
                cfg[variable_name] = get_trial_config(trial, cfg[variable_name])
    return cfg


# %%
@hydra.main(
    config_path="my_configs/",
    config_name="cartpole_wandb.yaml"
    # config_name="cartpole_wandb_optuna_choices.yaml"
)  # , version_base="1.3")
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        cfg_optuna = cfg_raw.optuna

        def objective(trial):
            cfg_sampled = get_trial_config(trial, cfg_raw.copy())

            logger = Logger(cfg_sampled)
            try:
                trial_result: float = run_dqn(cfg_sampled, logger, trial)
                logger.close()
                return trial_result
            except optuna.exceptions.TrialPruned:
                logger.close(exit_code=1)

        study = optuna.create_study(**cfg_optuna.study)
        study.optimize(func=objective, **cfg_optuna.optimize)

    else:
        logger = Logger(cfg_raw)
        run_dqn(cfg_raw, logger)


if __name__ == "__main__":
    main()
