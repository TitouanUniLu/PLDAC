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


#un des agents les plus importants:
#on utilise le ParallelGymAgent pour obtenir l'environment d'un agent puis on extrait les informations
#notamment l'image du cartpole a chaque temps t
#avec cette image, on la preprocess et on la passe dans un cnn (defini plus bas)
class AttributeAccessAgent(Agent):
    def __init__(self, env_agent, pre_processing_agent, cnn_agent):
        super().__init__()
        #les agents
        self.env_agent = env_agent  # ParallelGymAgent
        self.pre_processing_agent = pre_processing_agent
        self.cnn_agent = cnn_agent
        #init de la liste de listes pour store les images
        self.list_images = [[] for _ in range(self.env_agent.num_envs)]
        self.list_features = [[] for _ in range(self.env_agent.num_envs)]
        #print('env size: ', self.env_agent.num_envs)
    
    def forward(self, t: int, **kwargs):
        # Process for each environment
        for env_index in range(self.env_agent.num_envs):
            image = self.env_agent.envs[env_index].render()  # Retrieve the image
            image_pre_processed = self.pre_processing_agent.preprocess(image)  # Preprocess image
            self.list_images[env_index].append(image_pre_processed)  # Store preprocessed image

            features = self.cnn_agent.process_image(image_pre_processed)  # Process image to extract features
            self.list_features[env_index].append(features)

    def get_features(self, t):
        #On recupere les features pour chaque env a chaque temps t
        if t < len(self.list_features[0]):
            features = [self.list_features[env_index][t] for env_index in range(self.env_agent.num_envs)]
            # batch
            features_batch = torch.stack(features)
            # print(features_batch.shape)
            # print('Adding features to workspace at time ', t)            
            features = features_batch.squeeze(1)
            
            return features
        else:
            print(f"Features for timestep {t} pas dispo")
            return None


# agent pour gerer le preprocessing, rien de trop particulier
# on pourrait faire du preprocessing plus complexe si jamais on obtient des mauvais resultats
# faudrait reprendre celui de Mathis pour aller plus vite
# faudra aussi modifier la classe pour gerer plusieurs images (3 ou 4 d'un coup)
class PreProcessingAgent(Agent):
    def __init__(self):
        super().__init__()

    def preprocess(self, image):
        processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  #on mets en noir et blanc
        processed_image = cv2.resize(processed_image, (224, 224))  #on rescale 
        return processed_image #image de taille (224, 224)


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



# l'agent cnn qui implemente le modele cnn defini en haut
class CNNAgent(Agent):
    def __init__(self):
        super().__init__()
        self.cnn = CNN() #le cnn

    def process_image(self, image):
        #on converti l'image (preprocessed) en un tensor qu'on peut mettre dans le cnn
        #taille [1, 1, 224, 224], batch_size, channels, height, width
        image_tensor = torch.tensor(image, dtype=torch.float).unsqueeze(0).unsqueeze(0)  
        image_tensor = image_tensor / 255.0   #petite normalisation

        with torch.no_grad(): #on calcule pas le gradient (a voir si on le fait apres)
            features = self.cnn(image_tensor)
        features = features.cpu().numpy()
        return torch.tensor(features) #on renvoie le vecteur



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
    def __init__(self, input_dim, hidden_layers, action_dim, attribute_access_agent):
        super().__init__()
        self.model = build_mlp(
            [input_dim] + list(hidden_layers) + [action_dim], 
            activation=nn.ReLU()
        )
        self.attribute_access_agent = attribute_access_agent
        #print('num envs discreteqagent: ', self.attribute_access_agent.env_agent.num_envs)
        #print(self.attribute_access_agent)

    def forward(self, t: int, choose_action=True, **kwargs):
        # Retrieve the current feature vector for time step t from AttributeAccessAgent
        current_features = self.attribute_access_agent.get_features(t)
        self.set(('env/features', t), current_features)
        #print("obs ", current_features)
        
        q_values = self.model(current_features)
        self.set(("q_values", t), q_values)
        #print('q values: ', q_values)

        if choose_action:
            action = q_values.argmax(dim=1)
            #print('action: ', action)
            #print('action shape: ', action.size()[0])
            self.set(("action", t), action)


class EGreedyActionSelector(Agent):
    def __init__(self, epsilon):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, t: int, **kwargs):
        q_values = self.get(("q_values", t))
        #print(q_values)
        #print(q_values.size())
        size, nb_actions = q_values.size()

        # Flag 
        is_random = torch.rand(size).lt(self.epsilon).float()
        random_action = torch.randint(low=0, high=nb_actions, size=(size,))
        max_action = q_values.max(1)[1]

        # Choose the action based on the is_random flag
        action = is_random * random_action + (1 - is_random) * max_action

        # Sets the action at time t
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


# +
# For the tensor dimensions
# T = maximum number of time steps
# B = number of episodes run in parallel 
# A = state space dimension

def compute_critic_loss(cfg, reward: torch.Tensor, must_bootstrap: torch.Tensor, q_values: torch.Tensor, action: torch.LongTensor):
    q_values_for_actions = q_values.gather(2, action.unsqueeze(-1)).squeeze(-1)
    
    # Compute the max Q-value for the next state, but not for the last timestep
    next_q_values = q_values[1:].max(dim=2)[0]
    # Compute the expected Q-values (target) for the current state and action
    target_q_values = reward[:-1] + cfg["algorithm"]["discount_factor"] * next_q_values * must_bootstrap[:-1]
    
    # Compute the loss as the mean squared error between the current and target Q-values
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

def create_dqn_agent(cfg, train_env_agent, eval_env_agent):
    pre_processing_agent = PreProcessingAgent()
    cnn_agent = CNNAgent()

    attribute_access_train = AttributeAccessAgent(train_env_agent, pre_processing_agent, cnn_agent)
    attribute_access_eval = AttributeAccessAgent(eval_env_agent, pre_processing_agent, cnn_agent)


    critic = DiscreteQAgent(TENSRSIZE, cfg.algorithm.architecture.hidden_size, 2, attribute_access_train)

    #training
    q_agent = TemporalAgent(critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon)
    tr_agent = Agents(train_env_agent, attribute_access_train, critic, explorer)
    train_agent = TemporalAgent(tr_agent)

    #eval
    ev_agent = Agents(eval_env_agent, attribute_access_eval, critic)
    eval_agent = TemporalAgent(ev_agent)
    #print("success create_dqn_agent")
    return train_agent, eval_agent, q_agent

def create_best_dqn_agent(cfg, train_env_agent, eval_env_agent):
    pre_processing_agent = PreProcessingAgent()
    cnn_agent = CNNAgent()

    attribute_access_train = AttributeAccessAgent(train_env_agent, pre_processing_agent, cnn_agent)
    attribute_access_eval = AttributeAccessAgent(eval_env_agent, pre_processing_agent, cnn_agent)


    critic = DiscreteQAgent(TENSRSIZE, cfg.algorithm.architecture.hidden_size, 2, attribute_access_train)
    target_critic = copy.deepcopy(critic)
    target_q_agent = TemporalAgent(target_critic)

    #training
    q_agent = TemporalAgent(critic)
    explorer = EGreedyActionSelector(cfg.algorithm.epsilon)
    tr_agent = Agents(train_env_agent, attribute_access_train, critic, explorer)
    train_agent = TemporalAgent(tr_agent)

    #eval
    ev_agent = Agents(eval_env_agent, attribute_access_eval, critic)
    eval_agent = TemporalAgent(ev_agent)
    #print("success create_dqn_agent")
    return train_agent, eval_agent, q_agent, target_q_agent

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
    
    while nb_measures < 200: #cfg.algorithm.nb_measures
        train_workspace = Workspace()
        
        # default_action = torch.zeros(cfg.algorithm.n_envs, dtype=torch.long)
        # print(default_action)
        # train_workspace.set("action", 0, default_action)
        # print('current workspace action space')
        # print(train_workspace['action'])
        # print(train_workspace["action"].shape)
        
        # Run 
        train_agent(train_workspace, t=0, stop_variable="env/done", stochastic=True)
        
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
    "log_dir": "./tblogs/rl-cnn-" + str(time.time()),
    "cache_size": 10000,
    "every_n_seconds": 10,
    "verbose": False,    
    },

  "algorithm":{
    "seed": 3,
    "max_grad_norm": 0.5,
    "epsilon": 0.02,
    "n_envs": 8,
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

import sys
import os
import os.path as osp

path = os.getcwd()
print(f"Launch tensorboard from the shell:\n{osp.dirname(sys.executable)}/tensorboard --logdir={path}/tblogs")

cfg=OmegaConf.create(params)
torch.manual_seed(cfg.algorithm.seed)
train_agent, eval_agent, q_agent = run_dqn(cfg)

# Methode pour set une nouvelle variable dans le workspace, ici c'est juste un exemple avec un tensor 
# workspace.set('env/images', 0, torch.tensor((1,2)))
# print(workspace['env/images'])
# env_agent = ParallelGymAgent(partial(make_env, cfg.gym_env.env_name, render_mode="rgb_array", autoreset=False), cfg.algorithm.n_envs, reward_at_t=False)

# Notes et Remarques:
# -tester nouvelle fonction run_best_dqn (issu du notebook #2)
# -sauvegarder les features et images dans le workspace plutot qu'avec un autre agent qui passe les valeurs
# -ameliorer pre processing -> choper plusieurs images a la suite (implementer celui de mathis du coup)
# -besoin de changer le cnn? (taille de l'output surtout)