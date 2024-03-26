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
        self.cnn = CNN()  
    
    def forward(self, t: int, **kwargs):
        images = []
        features = []
        for env_index in range(self.env_agent.num_envs):
            # Récupérer l'image rendue
            image = self.env_agent.envs[env_index].render()
            #temporary preprocessing wont be needed probably
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #on mets en noir et blanc
            processed_image = cv2.resize(processed_image, (224, 224))
            images.append(torch.tensor(processed_image))
            #plt.imshow(processed_image)
            #plt.show()

            image_tensor = torch.tensor(processed_image, dtype=torch.float).unsqueeze(0).unsqueeze(0)  
            image_tensor = image_tensor / 255.0   #petite normalisation

            with torch.no_grad(): #on calcule pas le gradient (a voir si on le fait apres)
                feature = self.cnn(image_tensor)
            feature = feature.cpu().numpy()
            features.append(torch.tensor(feature))
        
        # Empiler toutes les images et les features dans des tensors
        features_tensor = torch.stack(features).squeeze(1)
        images_tensor = torch.stack(images)
        
        #ATTENTION: ON NE PEUT PAS STOCKER LES IMAGES DANS LE WORKSPACE ACTUELLEMENT, CA PRENDS BCP TROP DE PLACE
        #SI ON VEUT AJOUTER ON A UNE ERREUR -> RuntimeError: [enforce fail at alloc_cpu.cpp:114] data. DefaultCPUAllocator: not enough memory
        #self.set(("env/images", t), images_tensor) 
        # Ajouter les tensors dans le workspace
        self.set(("env/features", t), features_tensor)


#modele CNN: c'est une implementation plutot basqiue de cnn, truc classique qu'on trouve sur internet
#on fait avec 5 layers pcq a priori ca devrait suffire
#c'est possible qu'on ai un probleme ici a cause d'une grosse perte d'information (meme apres 100k pas la reward change pas)
TENSRSIZE = 32
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            #layer 1
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 16 x 112 x 112
            
            #layer 2
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 32 x 56 x 56
            
            #layer 3
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 64 x 28 x 28
            
            #layer 4
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 128 x 14 x 14
            
            #layer 5
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 256 x 7 x 7
        )
        
        self.fc_layers = nn.Sequential( #on cherche a avoir un vecteur de taille TENSRSIZE (une ligne)
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, TENSRSIZE),  
        )
        
    def forward(self, x): #ici c'est l'image qui a ete converti en tensor
        x = self.conv_layers(x)
        x = self.fc_layers(x)
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
        
        q_values = self.model(current_features)
        self.set(("q_values", t), q_values)

        if choose_action:
            action = q_values.argmax(dim=1)
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

# Configure the optimizer over the q agent
def setup_optimizer(cfg, q_agent):
    optimizer_args = get_arguments(cfg.optimizer)
    parameters = q_agent.parameters()
    optimizer = get_class(cfg.optimizer)(parameters, **optimizer_args)
    return optimizer


def get_env_agents(cfg):
    #print(cfg.algorithm.n_envs)
    train_env_agent = ParallelGymAgent(partial(make_env, cfg.gym_env.env_name, render_mode="rgb_array", autoreset=False), 
                                        cfg.algorithm.n_envs).seed(cfg.algorithm.seed) #cfg.algorithm.n_envs
    eval_env_agent = ParallelGymAgent(partial(make_env, cfg.gym_env.env_name, render_mode="rgb_array"), 
                                        cfg.algorithm.n_envs).seed(cfg.algorithm.seed)
    #print("success get_env_agents")
    return train_env_agent, eval_env_agent

def create_best_dqn_agent(cfg, train_env_agent, eval_env_agent):
    image_agent = ImageAgent(train_env_agent)

    critic = DiscreteQAgent(TENSRSIZE, cfg.algorithm.architecture.hidden_size, 2)
    target_critic = copy.deepcopy(critic)
    target_q_agent = TemporalAgent(target_critic)

    #training
    q_agent = TemporalAgent(critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon)
    tr_agent = Agents(train_env_agent, image_agent, critic, explorer)
    train_agent = TemporalAgent(tr_agent)

    #eval
    ev_agent = Agents(eval_env_agent, image_agent, critic)
    eval_agent = TemporalAgent(ev_agent)
    return train_agent, eval_agent, q_agent, target_q_agent

def run_best_dqn(cfg, compute_critic_loss):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = float('-inf')

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
                train_workspace, t=1, n_steps=cfg.algorithm.n_steps, stochastic=True
            )
        else:
            train_agent(
                train_workspace, t=0, n_steps=cfg.algorithm.n_steps, stochastic=True
            )

        # Get the transitions
        transition_workspace = train_workspace.get_transitions()

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
                    cfg, reward, must_bootstrap, q_values, target_q_values[1], action
                )
                # Store the loss for tensorboard display
                logger.add_log("critic_loss", critic_loss, nb_steps)

                optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_agent.parameters(), cfg.algorithm.max_grad_norm)
                optimizer.step()
                if nb_steps - last_critic_update_step > cfg.algorithm.target_critic_update:
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
                best_agent = copy.deepcopy(eval_agent.agent.agents[1])
                directory = "./dqn_critic/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "dqn0_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)

    return best_agent

new_params={
  "save_best": False,
  "logger":{
    "classname": "bbrl.utils.logger.TFLogger",
    "log_dir": "../tblogs/dqn-buffer-" + str(time.time()),
    "cache_size": 10000,
    "every_n_seconds": 10,
    "verbose": False,    
    },

  "algorithm":{
    "seed": 4,
    "max_grad_norm": 0.5,
    "epsilon": 0.02,
    "n_envs": 8,
    "n_steps": 32,
    "n_updates": 32,
    "eval_interval": 2000,
    "learning_starts": 2000,
    "nb_evals": 10,
    "buffer_size": 1e6,
    "batch_size": 256,
    "target_critic_update": 5000,
    "max_epochs": 3500,
    "discount_factor": 0.99,
    "architecture":{"hidden_size": [128, 128]},
  },
  "gym_env":{
    "env_name": "CartPole-v1",
  },
  "optimizer":
  {
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
  }
}

# +
import sys
import os
import os.path as osp
import gymnasium
from gymnasium import register

path = os.getcwd()
print(f"Launch tensorboard from the shell:\n{osp.dirname(sys.executable)}/tensorboard --logdir={path}/tblogs")

cartpole_spec = gymnasium.spec("CartPole-v1")
register(
    id="CartPole-v1",
    entry_point="cartpole:CartPoleEnv",
    max_episode_steps=cartpole_spec.max_episode_steps,
    reward_threshold=cartpole_spec.reward_threshold,
)
cfg=OmegaConf.create(new_params)
torch.manual_seed(cfg.algorithm.seed)

print('Looking for best agent')
best_agent = run_best_dqn(cfg, compute_critic_loss)

env = make_env(cfg.gym_env.env_name, render_mode="rgb_array")
record_video(env, best_agent, "videos/dqn-full.mp4")
video_display("videos/dqn-full.mp4")

# Notes et Remarques:
# -ameliorer pre processing -> choper plusieurs images a la suite (implementer celui de mathis du coup)
# -besoin de changer le cnn? (taille de l'output surtout)