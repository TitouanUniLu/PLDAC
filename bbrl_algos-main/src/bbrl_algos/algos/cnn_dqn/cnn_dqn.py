# General imports
import os
import os.path as osp
import sys
from pathlib import Path
from easypip import easyimport, is_notebook
from moviepy.editor import ipython_display as video_display
import time
from tqdm.auto import tqdm
from functools import partial
from omegaconf import DictConfig
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np
import hydra

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# BBRL imports
from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymnasium import ParallelGymAgent, make_env
from bbrl.utils.replay_buffer import ReplayBuffer


class ImageAgent(Agent):
    def __init__(self, env_agent):
        super().__init__()
        self.env_agent = env_agent
        self.cnn = SimpleCNN()
        self.image_buffer = [
            [torch.zeros(84, 84) for _ in range(4)]
            for _ in range(self.env_agent.num_envs)
        ]

    def forward(self, t: int, **kwargs):
        features = []
        total_loss = 0.0
        for env_index in range(self.env_agent.num_envs):
            image = self.env_agent.envs[env_index].render()
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.resize(
                processed_image, (84, 84)
            )  
            #plt.imshow(processed_image)
            #plt.show()
            processed_image_tensor = torch.tensor(processed_image, dtype=float)

            self.image_buffer[env_index].pop(0)
            self.image_buffer[env_index].append(processed_image_tensor)

            stacked_frames = np.stack(self.image_buffer[env_index], axis=0)
            input_tensor = (
                torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0) / 255.0
            )

            self.cnn.train()  # Set the CNN to training mode

            cnn_output = self.cnn(input_tensor).squeeze(0)

            real_observations = self.get(("env/env_obs", t))[env_index]

            loss = F.mse_loss(cnn_output, real_observations)  # Compute the MSE loss
            total_loss += loss.item()

            features.append(cnn_output)
        # Backpropagation
        #print(total_loss)
        total_loss_tensor = torch.tensor(total_loss, requires_grad=True)
        self.cnn.optimizer.zero_grad()
        total_loss_tensor.backward()
        self.cnn.optimizer.step()    

        features_tensor = torch.stack(features)
        self.set(("env/features", t), features_tensor)

TENSRSIZE = 4
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.conv1 = nn.Conv2d(4, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)        
        self.fc1 = nn.Linear(16 * 18 * 18, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))        
        x = x.view(-1, 16 * 18 * 18)  
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)


class DiscreteQAgent(Agent):
    def __init__(self, input_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [input_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )

    def forward(self, t: int, choose_action=True, **kwargs):
        current_features = self.get(("env/env_obs", t)) #"env/features"
        # print('features', current_features)
        # print(current_features.shape)

        q_values = self.model(current_features)
        self.set(("q_values", t), q_values)

        if choose_action:
            action = q_values.argmax(dim=1)
            # print('action', action)
            # print(action.shape)
            self.set(("action", t), action)


class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t: int, **kwargs):
        q_values = self.get(("q_values", t))
        size, nb_actions = q_values.size()

        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]

        action = is_random * random_action + (1 - is_random) * max_action

        self.set(("action", t), action.long())
        self.epsilon = max(0.001, self.epsilon * 0.995)


class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)

    def add_log(self, log_string, loss, steps):
        self.logger.add_scalar(log_string, loss.item(), steps)

    # A specific function for RL algorithms having a critic, an actor and an entropy losses
    def log_losses(self, critic_loss, entropy_loss, actor_loss, steps):
        self.add_log("critic_loss", critic_loss, steps)
        self.add_log("entropy_loss", entropy_loss, steps)
        self.add_log("actor_loss", actor_loss, steps)

    def log_reward_losses(self, rewards, nb_steps):
        self.add_log("reward/mean", rewards.mean(), nb_steps)
        self.add_log("reward/max", rewards.max(), nb_steps)
        self.add_log("reward/min", rewards.min(), nb_steps)
        self.add_log("reward/median", rewards.median(), nb_steps)


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
    qvals = q_values[0].gather(dim=1, index=act)
    qvals = qvals.squeeze(dim=1)
    return nn.MSELoss()(qvals, target)


# Configure the optimizer over the q agent
def setup_optimizer(cfg, q_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = q_agent.parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def get_env_agents(cfg):
    # print(cfg.algorithm.n_envs)
    train_env_agent = ParallelGymAgent(
        partial(
            make_env, cfg.gym_env.env_name, render_mode="rgb_array", autoreset=True
        ),
        cfg.algorithm.n_envs,
    ).seed(
        cfg.algorithm.seed
    )  # cfg.algorithm.n_envs
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, render_mode="rgb_array"),
        cfg.algorithm.n_envs,
    ).seed(cfg.algorithm.seed)
    # print("success get_env_agents")
    return train_env_agent, eval_env_agent


def create_best_dqn_agent(cfg, train_env_agent, eval_env_agent):
    image_train_agent = ImageAgent(train_env_agent)
    image_eval_agent = ImageAgent(eval_env_agent)

    critic = DiscreteQAgent(TENSRSIZE, cfg.algorithm.architecture.hidden_size, 2)
    target_critic = copy.deepcopy(critic)
    target_q_agent = TemporalAgent(target_critic)

    # training
    q_agent = TemporalAgent(critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon)
    tr_agent = Agents(train_env_agent, image_train_agent, critic, explorer)
    train_agent = TemporalAgent(tr_agent)

    # eval
    ev_agent = Agents(eval_env_agent, image_eval_agent, critic)
    eval_agent = TemporalAgent(ev_agent)
    return train_agent, eval_agent, q_agent, target_q_agent


def run_best_dqn(cfg, compute_critic_loss):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = float("-inf")

    # 2) Create the environment agents
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the DQN-like Agent
    train_agent, eval_agent, q_agent, target_q_agent = create_best_dqn_agent(
        cfg, train_env_agent, eval_env_agent
    )

    # 5) Configure the workspace to the right dimension
    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the agent() and critic_agent()
    # will take the workspace as parameter
    train_workspace = Workspace()  # Used for training
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # 6) Configure the optimizer over the dqn agent
    optimizer = setup_optimizer(cfg, q_agent)
    nb_steps = 0
    last_eval_step = 0
    last_critic_update_step = 0
    best_agent = eval_agent.agent.agents[1]

    # 7) Training loop
    pbar = tqdm(range(cfg.algorithm.max_epochs))
    for epoch in pbar:
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(
                train_workspace,
                t=1,
                n_steps=cfg.algorithm.n_steps_train,
                stochastic=True,
            )
        else:
            train_agent(
                train_workspace,
                t=0,
                n_steps=cfg.algorithm.n_steps_train,
                stochastic=True,
            )

        # Get the transitions

        transition_workspace = train_workspace.get_transitions()
        # print(Workspace.get(('env/features', epoch)))
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]

        # Adds the transitions to the workspace
        rb.put(transition_workspace)
        if rb.size() > cfg.algorithm.learning_starts:
            for _ in range(cfg.algorithm.n_updates):
                rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

                # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace)
                q_agent(rb_workspace, t=0, n_steps=2, choose_action=False)
                q_values, terminated, reward, action = rb_workspace[
                    "q_values", "env/terminated", "env/reward", "action"
                ]

                with torch.no_grad():
                    target_q_agent(rb_workspace, t=0, n_steps=2, stochastic=True)
                target_q_values = rb_workspace["q_values"]

                # Determines whether values of the critic should be propagated
                must_bootstrap = ~terminated[1]

                # Compute critic loss
                # FIXME: homogénéiser les notations (soit tranche temporelle, soit rien)
                critic_loss = compute_critic_loss(
                    cfg.algorithm.discount_factor,
                    reward,
                    must_bootstrap,
                    action,
                    q_values,
                    target_q_values[1],
                )
                # Store the loss for tensorboard display
                logger.add_log("critic_loss", critic_loss, nb_steps)

                optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    q_agent.parameters(), cfg.algorithm.max_grad_norm
                )
                optimizer.step()
                if (
                    nb_steps - last_critic_update_step
                    > cfg.algorithm.target_critic_update_interval
                ):
                    last_critic_update_step = nb_steps
                    target_q_agent.agent = copy.deepcopy(q_agent.agent)

        # Evaluate the current policy
        if nb_steps - last_eval_step > cfg.algorithm.eval_interval:
            last_eval_step = nb_steps
            eval_workspace = Workspace()
            eval_agent(
                eval_workspace, t=0, stop_variable="env/done", choose_action=True
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.log_reward_losses(rewards, nb_steps)
            pbar.set_description(f"nb steps: {nb_steps}, reward: {mean:.3f}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                best_agent = copy.deepcopy(eval_agent.agent.agents[2])
                directory = "./dqn_critic/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "dqn0_" + str(mean.item()) + ".agt"
                critic_agent = eval_agent.agent.agents[2]
                critic_agent.save_model(filename)

    return best_agent


# %%
@hydra.main(
    config_path="configs/",
    config_name="dqn_cartpole.yaml",
)
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed)
    run_best_dqn(cfg_raw, compute_critic_loss)


if __name__ == "__main__":
    main()
