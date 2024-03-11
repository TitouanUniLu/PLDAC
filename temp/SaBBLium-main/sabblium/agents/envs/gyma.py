#  SaBBLium ― A Python library for building and simulating multi-agent systems.
#
#  Copyright © Facebook, Inc. and its affiliates.
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
import warnings
from abc import ABC
from typing import (
    Any,
    Callable,
)


import torch
from gymnasium import Env, Space
from gymnasium.core import ActType, ObsType
from gymnasium.experimental.wrappers.numpy_to_torch import NumpyToTorchV0
from gymnasium.wrappers import AutoResetWrapper
from torch import nn, Tensor

from sabblium import SeedableAgent, SerializableAgent, TimeAgent

Frame = dict[str, Tensor]


def _torch_cat_dict(f: list[Frame]) -> Frame:
    # Auxiliary function to copy tensors for a specific key
    def copy_tensor(tensor_list: list[Frame], key: str):
        # Determine the shape and data type of the resulting tensor
        tensor_shape = (len(tensor_list),) + tuple(tensor_list[0][key].shape)
        tensor_dtype = tensor_list[0][key].dtype

        # Create an empty pre-allocated tensor with the appropriate shape and data type.
        result_tensor = torch.empty(tensor_shape, dtype=tensor_dtype)

        # Fill the pre-allocated tensor with the values from the input frames.
        for i, frame in enumerate(tensor_list):
            result_tensor[i].copy_(frame[key], non_blocking=True)

        # Return the filled tensor
        return result_tensor

    # For each key in the input frames, call the copy_tensor function to create and fill the pre-allocated tensor
    return {k: copy_tensor(f, k) for k in f[0]}


class GymAgent(TimeAgent, SeedableAgent, SerializableAgent, ABC):
    """Create an Agent from a gymnasium environment"""

    def __init__(
        self,
        input_string: str = "action",
        output_string: str = "env/",
        **kwargs,
    ):
        super().__init__(**kwargs)

        if self._seed is None:
            warnings.warn("The GymAgent won't be deterministic")

        self.ghost_params: nn.Parameter = nn.Parameter(torch.randn(()))

        self.input: str = input_string
        self.output: str = output_string
        self._timestep_from_reset: int = 0
        self._nb_reset: int = 0

        self.observation_space: Space[ObsType] | None = None
        self.action_space: Space[ActType] | None = None

    def forward(self, t: int, **kwargs) -> None:
        if t == 0:
            self._timestep_from_reset = 0
            self._nb_reset += 1
        else:
            self._timestep_from_reset += 0

    def set_state(self, states: Frame, t: int) -> None:
        for key, tensor in states.items():
            self.set(
                (self.output + key, t),
                tensor.to(self.ghost_params.device),
            )

    def get_observation_space(self) -> Space[ObsType]:
        """Return the observation space of the environment"""
        if self.observation_space is None:
            raise ValueError("The observation space is not defined")
        return self.observation_space

    def get_action_space(self) -> Space[ActType]:
        """Return the action space of the environment"""
        if self.action_space is None:
            raise ValueError("The action space is not defined")
        return self.action_space


class SerialGymAgent(GymAgent):
    """Create an Agent from a gymnasium environment
    To create an auto-reset SerialGymAgent, use the gymnasium `AutoResetWrapper` in the make_env_fn
    Warning: Make sure AutoResetWrapper is the outermost wrapper.
    """

    def __init__(
        self,
        make_env_fn: Callable[[dict[str, Any] | None], Env],
        num_envs: int = 1,
        make_env_args: dict[str, Any] | None = None,
        **kwargs,
    ):
        """Create an agent from a Gymnasium environment

        Args:
            make_env_fn ([function that returns a gymnasium.Env]): The function to create a single gymnasium environments
            num_envs ([int]): The number of environments to create, defaults to 1
            make_env_args (dict): The arguments of the function that creates a gymnasium.Env
            input_string (str, optional): [the name of the action variable in the workspace]. Defaults to "action".
            output_string (str, optional): [the output prefix of the environment]. Defaults to "env/".
        """
        super().__init__(**kwargs)
        if num_envs <= 0:
            raise ValueError("The number of environments must be > 0")

        self.make_env_fn: Callable[[dict[str, Any] | None], Env] = make_env_fn
        self.num_envs: int = num_envs

        self.envs: list[Env] = []
        self._cumulated_reward: Tensor = torch.zeros(num_envs, dtype=torch.float32)
        self._timestep: Tensor = torch.zeros(num_envs, dtype=torch.long)

        self._is_autoreset: bool = False
        self._last_frame: dict[int, Frame] = {}

        args: dict[str, Any] = make_env_args if make_env_args is not None else {}
        self._initialize_envs(num_envs=num_envs, make_env_args=args)

    def _initialize_envs(self, num_envs: int, make_env_args: dict[str, Any]):
        self.envs = [
            NumpyToTorchV0(self.make_env_fn(**make_env_args)) for _ in range(num_envs)
        ]
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        if type(self.envs[0].env) == AutoResetWrapper:
            self._is_autoreset = True

    def _reset(self, k: int) -> Frame:
        env: Env = self.envs[k]
        self._cumulated_reward[k] = 0.0

        observation, info = env.reset(
            seed=self._seed + k if self._seed is not None else None
        )

        self._timestep[k] = 0

        ret: Frame = {
            "obs": observation,
            "stopped": torch.tensor(False),
            "terminated": torch.tensor(False),
            "truncated": torch.tensor(False),
            "reward": torch.tensor(0.0),
            "cumulated_reward": self._cumulated_reward[k],
            "timestep": self._timestep[k],
            **{"info/" + k: v for k, v in info.items()},
        }

        self._last_frame[k] = ret
        return ret

    def _step(self, k: int, action: ActType):
        env = self.envs[k]

        # Following part to simplify when gymnasium AutoResetWrapper
        # will switch to Sutton & Barto’s notation.
        if (not self._is_autoreset) or (not self._last_frame[k]["stopped"]):
            observation, reward, terminated, truncated, info = env.step(action)
        else:
            observation = self.first_obs
            info = self.first_info
            terminated, truncated, reward = False, False, 0.0
        if self._is_autoreset and (terminated or truncated):
            # swap the observation and info
            observation, self.first_obs = info.pop("final_observation"), observation
            self.first_info = copy.copy(info)
            info = self.first_info.pop("final_info")
        # End of part to simplify.

        self._cumulated_reward[k] += reward
        self._timestep[k] += 1

        ret: Frame = {
            "obs": observation,
            "stopped": torch.tensor(terminated or truncated),
            "terminated": torch.tensor(terminated),
            "truncated": torch.tensor(truncated),
            "reward": torch.tensor(reward),
            "cumulated_reward": self._cumulated_reward[k],
            "timestep": self._timestep[k],
            **{"info/" + k: v for k, v in info.items()},
        }

        self._last_frame[k] = ret
        return ret

    def forward(self, t: int = 0, **kwargs) -> None:
        """Do one step by reading the `input_string` in the workspace at t-1
        If t==0, environments are reset with the seed given in the constructor
        Without seed provided, the environment is reset with a random seed.
        """
        super().forward(t, **kwargs)

        states = []
        if t == 0:
            for k, env in enumerate(self.envs):
                states.append(self._reset(k))
        else:
            action = self.get((self.input, t - 1))
            assert action.size()[0] == self.num_envs, "Incompatible number of envs"

            for k, env in enumerate(self.envs):
                if self._is_autoreset or not self._last_frame[k]["stopped"]:
                    states.append(self._step(k, action[k]))
                else:
                    states.append(self._last_frame[k])
        self.set_state(states=_torch_cat_dict(states), t=t)


class VisualGymAgent(GymAgent, ABC):
    """
    GymAgent compatible with visual observations
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def serialize(self):
        """Return a serializable GymAgent without the environments"""
        # TODO: add gymnasium environments to the serialization but not their unserializable attributes
        copied_agent = copy.copy(self)
        copied_agent.envs = None
        return copied_agent


class VisualSerialGymAgent(SerialGymAgent, VisualGymAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
