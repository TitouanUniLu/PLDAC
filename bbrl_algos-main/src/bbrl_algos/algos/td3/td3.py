import os
import copy
import numpy

import torch
import torch.nn as nn
import hydra
import optuna

from omegaconf import DictConfig
from bbrl.utils.chrono import Chrono

from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent, PrintAgent

from bbrl_algos.models.loggers import Logger
from bbrl.utils.replay_buffer import ReplayBuffer

from bbrl_algos.models.actors import ContinuousDeterministicActor
from bbrl_algos.models.critics import ContinuousQAgent
from bbrl_algos.models.shared_models import soft_update_params
from bbrl_algos.models.exploration_agents import AddGaussianNoise
from bbrl_algos.models.envs import get_env_agents
from bbrl_algos.models.hyper_params import launch_optuna
from bbrl_algos.models.utils import save_best

from bbrl.visu.plot_policies import plot_policy
from bbrl.visu.plot_critics import plot_critic

# HYDRA_FULL_ERROR = 1

import matplotlib

matplotlib.use("TkAgg")


# Create the TD3 Agent
def create_td3_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    actor = ContinuousDeterministicActor(
        obs_size,
        cfg.algorithm.architecture.actor_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.act,
    )
    # target_actor = copy.deepcopy(actor)
    noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)
    tr_agent = Agents(train_env_agent, actor, noise_agent)
    ev_agent = Agents(eval_env_agent, actor)

    critic_1 = ContinuousQAgent(
        obs_size,
        cfg.algorithm.architecture.critic_hidden_size,
        act_size,
        seed=cfg.algorithm.seed.q,
    )
    target_critic_1 = copy.deepcopy(critic_1).set_name("target-critic1")
    critic_2 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic_2 = copy.deepcopy(critic_2).set_name("target-critic2")

    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    return (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    )


# Configure the optimizer
def setup_optimizers(cfg, actor, critic_1, critic_2):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = nn.Sequential(critic_1, critic_2).parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, q_values_1, q_values_2, q_next):
    # Compute temporal difference
    target = (
        reward[:-1][0] + cfg.algorithm.discount_factor * q_next * must_bootstrap.int()
    )
    td_1 = target - q_values_1.squeeze(-1)
    td_2 = target - q_values_2.squeeze(-1)
    td_error_1 = td_1**2
    td_error_2 = td_2**2
    critic_loss_1 = td_error_1.mean()
    critic_loss_2 = td_error_2.mean()
    return critic_loss_1, critic_loss_2


def compute_actor_loss(q_values):
    actor_loss = -q_values
    return actor_loss.mean()


def run_td3(cfg, logger, trial=None):
    # 1)  Build the  logger
    best_reward = float("-inf")
    delta_list = []
    mean = 0

    # 2) Create the environment agents
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the TD3 Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    ) = create_td3_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent_1 = TemporalAgent(critic_1)
    target_q_agent_1 = TemporalAgent(target_critic_1)
    q_agent_2 = TemporalAgent(critic_2)
    target_q_agent_2 = TemporalAgent(target_critic_2)
    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic_1, critic_2)
    nb_steps = 0
    tmp_steps = 0

    # Training loop
    while nb_steps < cfg.algorithm.n_steps:
        # Execute the agent in the workspace
        if nb_steps > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)

            # below, was n_steps=cfg.algorithm.n_steps_train - 1
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps_train)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps_train)

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        if nb_steps > 0 or cfg.algorithm.n_steps_train > 1:
            rb.put(transition_workspace)

        for _ in range(cfg.algorithm.optim_n_updates):

            if nb_steps > cfg.algorithm.learning_starts:
                rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

                terminated, reward = rb_workspace["env/terminated", "env/reward"]
                # Determines whether values of the critic should be propagated
                # True if the task was not terminated
                must_bootstrap = ~terminated[1]

                # Critic update
                # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
                q_agent_1(rb_workspace, t=0, n_steps=1)
                q_values_rb_1 = rb_workspace["critic/q_values"]
                q_agent_2(rb_workspace, t=0, n_steps=1)
                q_values_rb_2 = rb_workspace["critic/q_values"]

                with torch.no_grad():
                    # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                    ag_actor(rb_workspace, t=1, n_steps=1)
                    # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                    target_q_agent_1(rb_workspace, t=1, n_steps=1)
                    post_q_values_1 = rb_workspace["target-critic1/q_values"]
                    target_q_agent_2(rb_workspace, t=1, n_steps=1)
                    post_q_values_2 = rb_workspace["target-critic2/q_values"]

                post_q_values = torch.min(post_q_values_1, post_q_values_2).squeeze(-1)
                # Compute critic loss
                critic_loss_1, critic_loss_2 = compute_critic_loss(
                    cfg,
                    reward,
                    must_bootstrap,
                    q_values_rb_1[0],
                    q_values_rb_2[0],
                    post_q_values[1],
                )
                logger.add_log("critic_loss_1", critic_loss_1, nb_steps)
                logger.add_log("critic_loss_2", critic_loss_2, nb_steps)
                critic_loss = critic_loss_1 + critic_loss_2

                # Critic update part #
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic_1.parameters(), cfg.algorithm.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    critic_2.parameters(), cfg.algorithm.max_grad_norm
                )
                critic_optimizer.step()

                # Actor update
                # Now we determine the actions the current policy would take in the states from the RB
                ag_actor(rb_workspace, t=0, n_steps=1)
                # We determine the Q values resulting from actions of the current policy
                # We arbitrarily chose to update the actor with respect to critic_1
                # and we back-propagate the corresponding loss to maximize the Q values
                q_agent_1(rb_workspace, t=0, n_steps=1)
                q_values_1 = rb_workspace["critic/q_values"]
                current_q_values = q_values_1.squeeze(-1)
                actor_loss = compute_actor_loss(current_q_values)
                logger.add_log("actor_loss", actor_loss, nb_steps)

                # Actor update part #
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), cfg.algorithm.max_grad_norm
                )
                actor_optimizer.step()

                # Soft update of target q function
                tau = cfg.algorithm.tau_target
                soft_update_params(critic_1, target_critic_1, tau)
                soft_update_params(critic_2, target_critic_2, tau)
                # soft_update_params(actor, target_actor, tau)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done")

            rewards = eval_workspace["env/cumulated_reward"][-1]
            q_agent_1(eval_workspace, t=0, stop_variable="env/done")
            q_values = eval_workspace["critic/q_values"].squeeze()
            delta = q_values - rewards
            maxi_delta = delta.max(axis=0)[0].detach().numpy()
            delta_list.append(maxi_delta)

            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)

            if mean > best_reward:
                best_reward = mean

            print(f"nb_steps: {nb_steps}, reward: {mean:.2f}, best: {best_reward:.2f}")

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if cfg.save_best and best_reward == mean:
                save_best(
                    eval_agent, cfg.gym_env.env_name, mean, "./td3_best_agents/", "td3"
                )

                if cfg.plot_agents:
                    plot_policy(
                        actor,
                        eval_env_agent,
                        "./td3_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        q_agent_1.agent,  # TODO: do we want to plot both critics?
                        eval_env_agent,
                        "./td3_plots/",
                        cfg.gym_env.env_name,
                        nb_steps,
                    )
    return best_reward


# %%
@hydra.main(
    config_path="configs/",
    # config_name="td3_cartpolecontinuous.yaml"
    # config_name="td3_pendulum_optuna.yaml",
    config_name="td3_swimmer.yaml",
    # config_name="td3_walker_test.yaml",
)  # , version_base="1.3")
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_td3)
    else:
        logger = Logger(cfg_raw)
        run_td3(cfg_raw, logger)


if __name__ == "__main__":
    main()
