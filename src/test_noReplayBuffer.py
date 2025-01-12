# General imports
import os
import os.path as osp
import sys
from pathlib import Path
from easypip import easyimport, easyinstall, is_notebook
from moviepy.editor import ipython_display as video_display
import time
from tqdm.auto import tqdm
from functools import partial
from omegaconf import OmegaConf
import copy
import matplotlib.pyplot as plt
import cv2
import numpy as np

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

#BBRL imports
from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymnasium import GymAgent, ParallelGymAgent, make_env, record_video
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.agents.gymnasium import make_env, GymAgent, ParallelGymAgent
from bbrl import instantiate_class


class ImageAgent(Agent):
    def __init__(self, env_agent):
        super().__init__()
        self.env_agent = env_agent
        self.cnn = SimpleCNN()  
        self.image_buffer = [[torch.zeros(84, 84) for _ in range(4)] for _ in range(self.env_agent.num_envs)]
    
    def forward(self, t: int, **kwargs):
        images = []
        features = []

        for env_index in range(self.env_agent.num_envs):
            
            image = self.env_agent.envs[env_index].render()

            # Temporary preprocessing (convert to grayscale and resize)
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            processed_image = cv2.resize(processed_image, (84, 84))  # Resize to match CNN input size

            processed_image_tensor = torch.tensor(processed_image, dtype=float)

            self.image_buffer[env_index].pop(0)
            self.image_buffer[env_index].append(processed_image_tensor)

            #if env_index == 0:
                #print(self.image_buffer[env_index])
                #print(len(self.image_buffer[env_index]))
                #print(processed_image_tensor)
                #print(processed_image_tensor.shape)
            
            stacked_frames = np.stack(self.image_buffer[env_index], axis=0)
    
            # Convert the numpy array to a PyTorch tensor and add a batch dimension
            # Also ensure the data type matches what PyTorch expects (float32 by default for CNNs)
            # Normalize the tensor to have values between 0 and 1 if it's not already done
            input_tensor = torch.tensor(stacked_frames, dtype=torch.float32).unsqueeze(0) / 255.0

            # Check if the input_tensor shape is [1, 4, 84, 84], where 1 is the batch size
            assert input_tensor.shape == (1, 4, 84, 84), f"Expected input_tensor shape to be [1, 4, 84, 84], got {input_tensor.shape}"

            self.cnn.eval()  # Comment out if you are in a training loop and the model is already in training mode
            with torch.no_grad():
                cnn_output = self.cnn(input_tensor)

            features.append(cnn_output.squeeze(0))
        features_tensor = torch.stack(features)

        self.set(("env/features", t), features_tensor)


TENSRSIZE = 4
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # Assuming the input images are grayscale, thus 1 channel
        # If they are colored images, you should change 1 to 3
        # Convolutional layer (sees 4x84x84 image tensor)
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4)
        # Convolutional layer (sees 20x20x16 tensor after pooling)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)
        # Fully connected layer (sees 9x9x32 tensor after pooling)
        self.fc1 = nn.Linear(in_features=32 * 9 * 9, out_features=256)
        # Output layer
        self.out = nn.Linear(in_features=256, out_features=TENSRSIZE)
        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Add sequence of convolutional and max pooling layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        # Flatten image input
        x = x.view(-1, 32 * 9 * 9)
        # Add dropout layer
        x = F.dropout(x, p=0.5, training=self.training)
        # Add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # Add output layer
        x = self.out(x)
        return x

#petite fonction pour afficher toutes les etapes d'une execution d'un agent dans un environement
def displayImagesPerAgent(images_per_agent):
    n_cols = 4  # nb de colonnes avec des images a afficher
    
    for env_index, images in enumerate(images_per_agent):
        print(f"Environment {env_index + 1}:")
        n_images = len(images)
        n_rows = (n_images + n_cols - 1) // n_cols  #nb de lignes

        #trucs a changer pour l'affichage mais pas beosin d'y toucher je pense
        figsize_width = 10  
        figsize_height = n_rows * (figsize_width / n_cols) * 0.5 
        plt.figure(figsize=(figsize_width, figsize_height))

        for i, image in enumerate(images):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow(image)
            plt.axis('off')  
        
        plt.tight_layout()
        plt.show()


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)

#QUAND UN BUILD LE MLP FAUT QUE sizes SOIT DU TYPE [128, 64, 2] par ex
#ex: mlp = build_mlp(sizes=[128] + [64, 64] + [2], activation=nn.ReLU(), output_activation=nn.Identity())


class DiscreteQAgent(Agent):
    def __init__(self, input_dim, hidden_layers, action_dim):
        super().__init__()
        self.model = build_mlp(
            [input_dim] + list(hidden_layers) + [action_dim], 
            activation=nn.ReLU()
        )

    def forward(self, t: int, choose_action=True, **kwargs):
        current_features = self.get(('env/features',t))
        #print('features', current_features)
        #print(current_features.shape)
        
        q_values = self.model(current_features)
        self.set(("q_values", t), q_values)

        if choose_action:
            action = q_values.argmax(dim=1)
            #print('action', action)
            #print(action.shape)
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

class Logger():

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

def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor, action: torch.LongTensor):
    q_values_for_actions = q_values.gather(2, action.unsqueeze(-1)).squeeze(-1)
    next_q_values = q_values[1:].max(dim=2)[0]
    target_q_values = reward[:-1] + cfg["algorithm"]["discount_factor"] * next_q_values * must_bootstrap[:-1]
    loss = F.mse_loss(q_values_for_actions[:-1], target_q_values)
    
    return loss

def setup_optimizer(cfg, q_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = q_agent.parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer

from typing import Tuple
def get_env_agents(cfg) -> Tuple[GymAgent, GymAgent]:
    # Returns a pair of environments (train / evaluation) based on a configuration `cfg`
    
    # Train environment
    train_env_agent = ParallelGymAgent(
        partial(make_env,  cfg.gym_env.env_name, render_mode="rgb_array", autoreset=False),
        cfg.algorithm.n_envs
    ).seed(cfg.algorithm.seed)

    # Test environment
    eval_env_agent = ParallelGymAgent(
        partial(make_env, cfg.gym_env.env_name, render_mode="rgb_array"), 
        cfg.algorithm.nb_evals
    ).seed(cfg.algorithm.seed)

    return train_env_agent, eval_env_agent

def create_dqn_agent(cfg, train_env_agent, eval_env_agent) -> Tuple[TemporalAgent, TemporalAgent]:
    # Get the observation / action state space dimensions
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    image_agent_train = ImageAgent(train_env_agent)
    image_agent_eval = ImageAgent(eval_env_agent)

    # Our discrete Q-Agent    
    critic = DiscreteQAgent(TENSRSIZE, cfg.algorithm.architecture.hidden_size, act_size)

    # The agent used for training
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon)
    q_agent = TemporalAgent(critic)
    tr_agent = Agents(train_env_agent, image_agent_train, critic, explorer)
    train_agent = TemporalAgent(tr_agent)

    # The agent used for evaluation
    ev_agent = Agents(eval_env_agent, image_agent_eval, critic)
    eval_agent = TemporalAgent(ev_agent)
    
    return train_agent, eval_agent, q_agent

def run_dqn(cfg):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = float('-inf')

    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)
    
    # 3) Create the DQN Agent
    train_agent, eval_agent, q_agent = create_dqn_agent(
        cfg, train_env_agent, eval_env_agent
    )

    # Note that no parameter is needed to create the workspace.
    # In the training loop, calling the train_agent
    # will take the workspace as parameter

    # 6) Configure the optimizer
    optimizer = setup_optimizer(cfg, q_agent)
    nb_steps = 0
    tmp_steps = 0
    nb_measures = 0

    while nb_measures < cfg.algorithm.nb_measures:
        train_workspace = Workspace()
        # Run 
        train_agent(train_workspace, t=0, stop_variable="env/done", stochastic=True)
        #print(train_workspace['action'])

        q_values, done, truncated, reward, action = train_workspace[
            "q_values", "env/done", "env/truncated", "env/reward", "action"
        ]

        nb_steps += len(action.flatten())
        
        # Determines whether values of the critic should be propagated
        # True if the episode reached a time limit or if the task was not done
        # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj
        must_bootstrap = torch.logical_or(~done, truncated)
        
        # Compute critic loss
        critic_loss = compute_critic_loss(cfg, reward, must_bootstrap, q_values, action)

        # Store the loss for tensorboard display
        logger.add_log("critic_loss", critic_loss, nb_steps)

        optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            q_agent.parameters(), cfg.algorithm.max_grad_norm
        )
        optimizer.step()

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            nb_measures += 1
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(
                eval_workspace, t=0, stop_variable="env/done", choose_action=True
            )
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./dqn_critic/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "dqn_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)
                
    return train_agent, eval_agent, q_agent

params={
  "save_best": False,
  "logger":{
    "classname": "bbrl.utils.logger.TFLogger",
    "log_dir": "./tblogs/dqn-simple-" + str(time.time()),
    "cache_size": 10000,
    "every_n_seconds": 3,
    "verbose": False,    
    },

  "algorithm":{
    "seed": 3,
    "max_grad_norm": 0.5,
    "epsilon": 0.02,
    "n_envs": 2,
    "n_steps": 32,
    "eval_interval": 2000,
    "nb_measures": 200,
    "nb_evals": 10,
    "discount_factor": 0.99,
    "architecture":{"hidden_size": [128, 128]},
  },
  "gym_env":{
    "env_name": "CartPole-v1",
  },
  "optimizer":
  {
    "classname": "torch.optim.Adam",
    "lr": 2e-3,
  }
}

# +
import sys
import os
import os.path as osp
import gymnasium
from gymnasium import register

cartpole_spec = gymnasium.spec("CartPole-v1")
register(
    id="CartPole-v1",
    entry_point="cartpole:CartPoleEnv",
    max_episode_steps=cartpole_spec.max_episode_steps,
    reward_threshold=cartpole_spec.reward_threshold,
)
config=OmegaConf.create(params)
torch.manual_seed(config.algorithm.seed)
train_agent, eval_agent, q_agent = run_dqn(config)

# Visualization
env = make_env(config.gym_env.env_name, render_mode="rgb_array")
record_video(env, train_agent.agent.agents[1], "videos/dqn-simple.mp4")
video_display("videos/dqn-simple.mp4")