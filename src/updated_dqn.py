#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
import os
import numpy as np
from typing import Callable, List
from PIL import Image

import hydra
import optuna
from omegaconf import DictConfig

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# %%
import gymnasium as gym
from gymnasium import Env
from gymnasium.wrappers import AutoResetWrapper

# %%
from bbrl import get_arguments, get_class
from bbrl.agents import TemporalAgent, Agents, PrintAgent
from bbrl.workspace import Workspace


from bbrl.visu.plot_critics import plot_discrete_q, plot_critic

from bbrl.utils.chrono import Chrono

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

from bbrl.agents.gymnasium import make_env, ParallelGymAgent
from functools import partial
from bbrl.agents import Agent, SeedableAgent, TimeAgent, SerializableAgent
from bbrl import instantiate_class
import yaml
import random
import cv2
import time
import mon_env

matplotlib.use("TkAgg")

class MazeMDPContinuousWrapper(gym.Wrapper):
    """
    Specific wrapper to turn the Tabular MazeMDP into a continuous state version
    """

    def __init__(self, env):
        super(MazeMDPContinuousWrapper, self).__init__(env)
        # Building a new continuous observation space from the coordinates of each state
        high = np.array(
            [
                env.coord_x.max() + 1,
                env.coord_y.max() + 1,
            ],
            dtype=np.float32,
        )
        low = np.array(
            [
                env.coord_x.min(),
                env.coord_y.min(),
            ],
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(low, high)
        # print("building maze:", high, low)

    def is_continuous_state(self):
        # By contrast with the wrapped environment where the state space is discrete
        return True

    def reset(self, **kwargs):
        obs, _ = self.env.reset(**kwargs)
        x = self.env.coord_x[obs]
        y = self.env.coord_y[obs]
        xc = x + random.random()
        yc = y + random.random()
        continuous_obs = [xc, yc]
        return np.array(continuous_obs, dtype=np.float32), {}

    def step(self, action):
        # Turn the discrete state into a pair of continuous coordinates
        # Take the coordinates of the state and add a random number to x and y to
        # sample anywhere in the [1, 1] cell...
        next_state, reward, terminated, truncated, info = self.env.step(action)
        x = self.env.coord_x[next_state]
        y = self.env.coord_y[next_state]
        # if reward > 0 : print("reward_found", x, y, reward)
        xc = x + random.random()
        yc = y + random.random()
        next_continuous = [xc, yc]
        if truncated or terminated:
            info = {"final_observation": next_continuous}
        return (
            np.array(next_continuous, dtype=np.float32),
            reward,
            terminated,
            truncated,
            info,
        )


def get_trial_value(trial: optuna.Trial, cfg: DictConfig, variable_name: str):
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

def launch_optuna(cfg_raw, run_func):
    cfg_optuna = cfg_raw.optuna

    def objective(trial):
        cfg_sampled = get_trial_config(trial, cfg_raw.copy())

        logger = Logger(cfg_sampled)
        try:
            trial_result: float = run_func(cfg_sampled, logger, trial)
            logger.close()
            return trial_result
        except optuna.exceptions.TrialPruned:
            logger.close()
            return float("-inf")

    study = hydra.utils.call(cfg_optuna.study)
    study.optimize(func=objective, **cfg_optuna.optimize)

    file = open("best_params.yaml", "w")
    yaml.dump(study.best_params, file)
    file.close()

def save_best(agent, env_name, score, dirname, fileroot):
    if not os.path.exists(dirname + env_name):
        os.makedirs(dirname + env_name)
    filename = dirname + env_name + fileroot + str(score.item()) + ".agt"
    agent.save_model(filename)

def build_alt_mlp(sizes, activation):
    layers = []
    for j in range(len(sizes) - 1):
        if j < len(sizes) - 2:
            layers += [nn.Linear(sizes[j], sizes[j + 1]), activation]
        else:
            layers += [nn.Linear(sizes[j], sizes[j + 1])]
    return nn.Sequential(*layers)

class Logger:
    def __init__(self, cfg):
        self.logger = instantiate_class(cfg.logger)
        self.logger.save_hps(cfg)

    def add_log(self, log_string, log_item, steps):
        if isinstance(log_item, torch.Tensor) and log_item.dim() == 0:
            log_item = log_item.item()
        self.logger.add_scalar(log_string, log_item, steps)

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
        self.add_log("reward/std", rewards.std(), nb_steps)

    def close(self) -> None:
        self.logger.close()


class NamedCritic(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(
        self,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name

    def set_name(self, name: str):
        self.name = name
        return self


class DiscreteQAgent(NamedCritic):
    def __init__(
        self,
        state_dim,
        hidden_layers,
        action_dim,
        name="critic",
        *args,
        **kwargs,
    ):
        super().__init__(name, *args, **kwargs)
        self.model = build_alt_mlp(
            [state_dim] + list(hidden_layers) + [action_dim], activation=nn.ReLU()
        )
        self.is_q_function = True

    def forward(self, t, choose_action=True, **kwargs):
        obs = self.get(("env/features", t))
        # print("in critic forward: obs:", obs)
        q_values = self.model(obs)
        self.set((f"{self.name}/q_values", t), q_values)
        # Sets the action
        if choose_action:
            action = q_values.argmax(1)
            self.set(("action", t), action)

    def predict_action(self, obs, stochastic):
        q_values = self.model(obs).squeeze(-1)
        if stochastic:
            probs = torch.softmax(q_values, dim=-1)
            action = torch.distributions.Categorical(probs).sample()
        else:
            action = q_values.argmax(-1)
        return action

    def predict_value(self, obs, action):
        q_values = self.model(obs)
        return q_values[action[0].int()]


class EGreedyActionSelector(TimeAgent, SeedableAgent, SerializableAgent):
    def __init__(self, epsilon, epsilon_end=None, epsilon_decay=None, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

    def decay(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_end)

    def forward(self, t, **kwargs):
        q_values = self.get(("critic/q_values", t))
        nb_actions = q_values.size()[1]
        size = q_values.size()[0]
        # TODO: make it deterministic if seeded
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]
        action = is_random * random_action + (1 - is_random) * max_action
        action = action.long()
        self.set(("action", t), action)

def local_get_env_agents(cfg):
    ''' ADDED render_mode="rgb_array" TO ParallelGymAgent FOR THE IMAGE AGENT'''
    eval_env_agent = ParallelGymAgent(
        partial(
            make_env,
            cfg.gym_env.env_name,
            render_mode="rgb_array",
            autoreset=False,
        ),
        cfg.algorithm.nb_evals,
        include_last_state=True,
        seed=cfg.algorithm.seed.eval,
    )
    train_env_agent = ParallelGymAgent(
        partial(
            make_env,
            cfg.gym_env.env_name,
            render_mode="rgb_array",
            autoreset=True,
        ),
        cfg.algorithm.n_envs,
        include_last_state=True,
        seed=cfg.algorithm.seed.train,
    )
    return train_env_agent, eval_env_agent


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
    qvals = q_values[0].gather(dim=1, index=act)
    qvals = qvals.squeeze(dim=1)
    return nn.MSELoss()(qvals, target)


# %%
def create_dqn_agent(cfg_algo, train_env_agent, eval_env_agent):
    # obs_space = train_env_agent.get_observation_space()
    # obs_shape = obs_space.shape if len(obs_space.shape) > 0 else obs_space.n

    # act_space = train_env_agent.get_action_space()
    # act_shape = act_space.shape if len(act_space.shape) > 0 else act_space.n

    state_dim, action_dim = train_env_agent.get_obs_and_actions_sizes()

    critic = DiscreteQAgent(
        state_dim=state_dim,
        hidden_layers=list(cfg_algo.architecture.hidden_sizes),
        action_dim=action_dim,
        seed=cfg_algo.seed.q,
    )

    explorer = EGreedyActionSelector(
        name="action_selector",
        epsilon=cfg_algo.explorer.epsilon_start,
        epsilon_end=cfg_algo.explorer.epsilon_end,
        epsilon_decay=cfg_algo.explorer.decay,
        seed=cfg_algo.seed.explorer,
    )
    q_agent = TemporalAgent(critic)

    train_image_agent = ImageAgent(train_env_agent)
    eval_image_agent = ImageAgent(eval_env_agent)

    ''' ADD IMAGE AGENTS HERE'''
    tr_agent = Agents(train_env_agent, train_image_agent, critic, explorer)  # , PrintAgent())
    ev_agent = Agents(eval_env_agent, eval_image_agent, critic)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)

    return train_agent, eval_agent, q_agent


# %%
# Configure the optimizer over the q agent
def setup_optimizer(optimizer_cfg, q_agent):
    optimizer_args = get_arguments(optimizer_cfg)
    parameters = q_agent.parameters()
    optimizer = get_class(optimizer_cfg)(parameters, **optimizer_args)
    return optimizer


# %%
def run_dqn(cfg, logger, trial=None):
    best_reward = float("-inf")
    if cfg.collect_stats:
        directory = "./dqn_data/"
        if not os.path.exists(directory):
            os.makedirs(directory)
        filename = directory + "dqn_" + cfg.gym_env.env_name + ".data"
        fo = open(filename, "wb")
        stats_data = []

    # 1) Create the environment agent
    train_env_agent, eval_env_agent = local_get_env_agents(cfg)

    # 2) Create the DQN-like Agent
    train_agent, eval_agent, q_agent = create_dqn_agent(
        cfg.algorithm, train_env_agent, eval_env_agent
    )

    # 3) Create the training workspace
    train_workspace = Workspace()  # Used for training

    # 5) Configure the optimizer
    optimizer = setup_optimizer(cfg.optimizer, q_agent)

    # 6) Define the steps counters
    nb_steps = 0
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

        # The q agent needs to be executed on the rb_workspace workspace (gradients are removed in workspace).
        q_agent(transition_workspace, t=0, n_steps=2, choose_action=False)

        q_values, terminated, reward, action = transition_workspace[
            "critic/q_values",
            "env/terminated",
            "env/reward",
            "action",
        ]

        # Determines whether values of the critic should be propagated
        # True if the task was not terminated.
        must_bootstrap = ~terminated

        critic_loss = compute_critic_loss(
            cfg.algorithm.discount_factor,
            reward,
            must_bootstrap,
            action,
            q_values,
        )

        # Store the loss
        logger.add_log("critic_loss", critic_loss, nb_steps)

        optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            q_agent.parameters(), cfg.algorithm.max_grad_norm
        )

        optimizer.step()

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
            logger.log_reward_losses(rewards, nb_steps)
            mean = rewards.mean()

            if mean > best_reward:
                best_reward = mean

            print(
                f"nb_steps: {nb_steps}, reward: {mean:.02f}, best: {best_reward:.02f}"
            )

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            if cfg.save_best and best_reward == mean:
                save_best(
                    eval_agent,
                    cfg.gym_env.env_name,
                    best_reward,
                    "./dqn_best_agents/",
                    "dqn",
                )
                if cfg.plot_agents:
                    critic = eval_agent.agent.agents[1]
                    plot_discrete_q(
                        critic,
                        eval_env_agent,
                        best_reward,
                        "./dqn_plots/",
                        cfg.gym_env.env_name,
                        input_action="policy",
                    )
                    plot_discrete_q(
                        critic,
                        eval_env_agent,
                        best_reward,
                        "./dqn_plots2/",
                        cfg.gym_env.env_name,
                        input_action=None,
                    )
            if cfg.collect_stats:
                stats_data.append(rewards)

            if trial is not None:
                trial.report(mean, nb_steps)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    if cfg.collect_stats:
        # All rewards, dimensions (# of evaluations x # of episodes)
        stats_data = torch.stack(stats_data, axis=-1)
        print(np.shape(stats_data))
        np.savetxt(filename, stats_data.numpy())
        fo.flush()
        fo.close()

    return best_reward


TENSRSIZE = 4 


# class SimpleCNN(nn.Module):  ### OLD CNN 
#     def __init__(self):
#         super(SimpleCNN, self).__init__()

#         self.conv1 = nn.Conv2d(4, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 16, 5)        
#         self.fc1 = nn.Linear(16 * 18 * 18, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 4)

#         self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))        
#         x = x.view(-1, 16 * 18 * 18)  
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x

# class ImageAgent(Agent): ### IMAGE AGENT W/ SIMULTANEOUSLY TRAINING CNN
#         def __init__(self, env_agent):
#             super().__init__()
#             self.env_agent = env_agent
#             self.cnn = SimpleCNN()
#             self.image_buffer = [
#                 [torch.zeros(84, 84) for _ in range(4)]
#                 for _ in range(self.env_agent.num_envs)
#             ]
    
#         def forward(self, t: int, **kwargs):
#             features = []
#             total_loss = 0.0
#             for env_index in range(self.env_agent.num_envs):
#                 image = self.env_agent.envs[env_index].render()
#                 processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
#                 processed_image = cv2.resize(
#                     processed_image, (84, 84)
#                 )  
#                 # plt.imshow(processed_image)
#                 # plt.show()
#                 # time.sleep(5)
                
#                 processed_image_tensor = torch.tensor(processed_image, dtype=float)
    
#                 self.image_buffer[env_index].pop(0)
#                 self.image_buffer[env_index].append(processed_image_tensor)
    
#                 stacked_frames = np.stack(self.image_buffer[env_index], axis=0)
#                 input_tensor = (
#                     torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0) / 255.0
#                 )
    
#                 self.cnn.train()  # Set the CNN to training mode
    
#                 cnn_output = self.cnn(input_tensor).squeeze(0)
    
#                 real_observations = self.get(("env/env_obs", t))[env_index]
    
#                 loss = F.mse_loss(cnn_output, real_observations)  # Compute the MSE loss
    
#                 # Backpropagation
#                 self.cnn.optimizer.zero_grad()
#                 loss.backward()
#                 self.cnn.optimizer.step()
    
#                 total_loss += loss.item()
#                 features.append(cnn_output)
            
#             # print(f' loss at time {t}: {total_loss}')
#             # total_loss_tensor = torch.tensor(total_loss, requires_grad=True)
                
    
#             features_tensor = torch.stack(features)
#             self.set(("env/features", t), features_tensor)


class CNN(nn.Module): ### CNN FOR RGB IMAGES 
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(16, 32, kernel_size=(3, 3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
        )
        # Correctly calculate the input size for the linear layer based on the output from conv_layers
        self.fc_layers = nn.Sequential(
            nn.Linear(int(63360/2), 128),  # Adjusted based on actual output size
            nn.ReLU(),
            nn.Linear(128, 4)  # Predicting 4 state variables
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = self.fc_layers(x)
        return x
    
class CartPoleCNN(nn.Module): ### CNN FOR GREY IMAGES
    def __init__(self):
        super(CartPoleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=5, stride=1, padding=2),  # Input: 4 gray images, output: 16 channels, 60x135
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Output size: ? 30x67
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # Output size: ? 15x33
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)                  # Output size: ? 7x16
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(64 * 7 * 16, 128),
            nn.ReLU(),
            nn.Linear(128,4)    # x, x_dot, theta, theta_dot
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # Flatten the output for the fully connected layers
        x = self.fc_layers(x)
        return x

# Transformation for images
transform_rgb = transforms.Compose([transforms.ToPILImage(), 
                    transforms.Resize(60, interpolation=Image.LANCZOS),
                    transforms.ToTensor()])

# Cart location for centering image crop
def get_cart_location(screen_width,env):
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

# Cropping, downsampling (and Grayscaling) image
def get_screen_rgb(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width,env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # Resize, and add a batch dimension (BCHW)
    screen = torch.from_numpy(screen)
    return transform_rgb(screen)

sequence_length = 2 # Modifier ce paramètre selon le nb d'images en entrée

class ImageAgent(Agent): ### Image Agent for RGB Images
    def __init__(self, env_agent, model_path = 
                 os.path.abspath('C:/Users/hatem/OneDrive/Documents/Programmation/M1-S2/PLDAC/PLDAC_BBRL/src/cartpole_cnn_rgb_enhanced_2_img.pth')
    
):
        super().__init__()
        self.env_agent = env_agent
        self.cnn = CNN()
        self.cnn.load_state_dict(torch.load(model_path))
        self.cnn.eval() #change to .train() pour la backprop

        # Initialize an image buffer for each environment, storing RGB images
        self.image_buffer = [
            [torch.zeros(3, 60, 135) for _ in range(sequence_length)] 
            for _ in range(self.env_agent.num_envs)
        ]

    def forward(self, t: int, **kwargs):
        features = []

        for env_index in range(self.env_agent.num_envs):
            env = self.env_agent.envs[env_index]
            processed_image = get_screen_rgb(env)
            
            # Update the image buffer
            self.image_buffer[env_index].pop(0)
            self.image_buffer[env_index].append(processed_image)

            # Stack and normalize the images
            input_tensor = torch.stack(self.image_buffer[env_index][-sequence_length :], dim=0).permute(1, 0, 2, 3).unsqueeze(0) # Shape [1, 3, sequence_length, 64, 64]
            # Perform inference
            with torch.no_grad():
                cnn_output = self.cnn(input_tensor).squeeze(0)

            # print("COMPARAISON :")
            # print(cnn_output, self.get(("env/env_obs",t))[env_index])

            features.append(cnn_output)

        features_tensor = torch.stack(features)
        self.set(("env/features", t), features_tensor)

transform_grey = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((60, 135)),
    transforms.Grayscale()
])


# Cropping, downsampling (and Grayscaling) image
def get_screen_grey(env):
    # Returned screen requested by gym is 400x600x3, but is sometimes larger
    # such as 800x1200x3. Transpose it into torch order (CHW).
    screen = env.render().transpose((2, 0, 1))
    # Cart is in the lower half, so strip off the top and bottom of the screen
    _, screen_height, screen_width = screen.shape
    screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
    view_width = int(screen_width * 0.6)
    cart_location = get_cart_location(screen_width,env)
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescale, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    # Resize, and add a batch dimension (BCHW)
    return transform_grey(screen.transpose(1,2,0)).squeeze(0)


# class ImageAgent(Agent): ### Image Agent for grey images
#     def __init__(self, env_agent, model_path = 
#                  os.path.abspath('C:/Users/hatem/OneDrive/Documents/Programmation/M1-S2/PLDAC/PLDAC_BBRL/src/cnn_grey_1000.pth')
    
# ):
#         super().__init__()
#         self.env_agent = env_agent
#         self.cnn = CartPoleCNN()
#         self.cnn.load_state_dict(torch.load(model_path))
#         self.cnn.eval() #change to .train() pour la backprop

#         # Initialize an image buffer for each environment, storing RGB images
#         self.image_buffer = [
#             [torch.zeros(60, 135) for _ in range(4)]
#             for _ in range(self.env_agent.num_envs)
#         ]

#     def forward(self, t: int, **kwargs):
#         features = []
  
#         for env_index in range(self.env_agent.num_envs):
#             env = self.env_agent.envs[env_index]
#             # a = env.render(mode="rgb_array")
#             # print(a.shape)

#             processed_image = get_screen_grey(env)
            
#             # Update the image buffer
#             self.image_buffer[env_index].pop(0)
#             self.image_buffer[env_index].append(processed_image)

#             # Stack and normalize the images
#             input_tensor = torch.stack(self.image_buffer[env_index][-4:], dim=0) # Shape [1, 3, sequence_length, 64, 64]

#             # Perform inference
#             with torch.no_grad():
#                 cnn_output = self.cnn(input_tensor.unsqueeze(0)).squeeze(0)

#             # print("COMPARAISON :")
#             # print(cnn_output, self.get(("env/env_obs",t))[env_index])

#             features.append(cnn_output)

#         features_tensor = torch.stack(features)
#         self.set(("env/features", t), features_tensor)


# %%
@hydra.main(
    config_path="configs/",
    # config_name="dqn_cartpole.yaml",
    config_name="dqn_lunar_lander.yaml",
)  # , version_base="1.3")
def main(cfg_raw: DictConfig):
    torch.random.manual_seed(seed=cfg_raw.algorithm.seed.torch)

    if "optuna" in cfg_raw:
        launch_optuna(cfg_raw, run_dqn)
    else:
        logger = Logger(cfg_raw)
        run_dqn(cfg_raw, logger)


if __name__ == "__main__":
    main()
