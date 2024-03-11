#  SaBBLium ― A Python library for building and simulating multi-agent systems.
#
#  Copyright © Facebook, Inc. and its affiliates.
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import copy
import pickle
import time
from abc import ABC
from typing import Any, IO, Self

from torch import Tensor
import torch.nn as nn

from .workspace import Workspace


class Agent(nn.Module, ABC):
    """An `Agent` is a `torch.nn.Module` that reads and writes into a `sabblium.Workspace`"""

    def __init__(self, name: str | None = None, **kwargs) -> None:
        """To create a new Agent

        Args:
            name ([str | None]): An agent can have a name that will allow to perform operations on agents that are composed into more complex agents.
        """
        super().__init__(**kwargs)
        self.workspace: Workspace | None = None
        self._name: str | None = name
        self.__trace_file: IO | None = None

    def set_name(self, name: str) -> None:
        """Set the name of this agent

        Args:
            name (str): The name
        """
        self._name = name

    def get_name(self) -> str | None:
        """Get the name of the agent

        Returns:
            str: the name
        """
        return self._name

    def set_trace_file(self, filename: str) -> None:
        print("[TRACE]: Tracing agent in file " + filename)
        self.__trace_file = open(filename, "wt")

    def __call__(self, workspace: Workspace, **kwargs) -> Any:
        """Execute an agent of a `sabblium.Workspace`

        Args:
            workspace (sabblium.Workspace): the workspace on which the agent operates.
        """
        if workspace is None:
            raise TypeError(
                "[{}.__call__] workspace must not be None".format(self.__name__)
            )
        self.workspace = workspace
        ret = self.forward(**kwargs)
        self.workspace = None
        return ret

    def forward(self, **kwargs) -> Any:
        """The generic function to override when defining a new agent"""
        raise NotImplementedError("Your agent must override forward")

    def clone(self) -> Self:
        """Create a clone of the agent

        Returns:
            sabblium.Agent: A clone
        """
        self.zero_grad()
        return copy.deepcopy(self)

    def get(self, index: str | tuple[str, int]) -> Tensor:
        """Returns the value of a particular variable in the agent workspace

        Args:
            index (str or tuple(str,int)): if str, returns the variable workspace[str].
            If tuple(var_name, t), returns workspace[var_name] at time t
        """
        if self.__trace_file is not None:
            t: float = time.time()
            self.__trace_file.write(
                str(self) + " type = " + str(type(self)) + " time = ",
                t,
                " get ",
                index,
                "\n",
            )
        if isinstance(index, str):
            return self.workspace.get_full(index)
        elif isinstance(index, tuple):
            return self.workspace.get(index[0], index[1])
        else:
            raise TypeError("index must be either str or tuple(str, int)".format())

    def get_time_truncated(self, var_name: str, from_time: int, to_time: int) -> Tensor:
        """Return a variable truncated between from_time and to_time"""
        return self.workspace.get_time_truncated(var_name, from_time, to_time)

    def set(self, index: str | tuple[str, int], value: Tensor) -> None:
        """Write a variable in the workspace

        Args:
            index (str or tuple(str,int)):
            value (torch.Tensor): the value to write.
        """
        if self.__trace_file is not None:
            t = time.time()
            self.__trace_file.write(
                str(self) + " type = " + str(type(self)) + " time = ",
                t,
                " set ",
                index,
                " = ",
                value.size(),
                "/",
                value.dtype,
                "\n",
            )
        if isinstance(index, str):
            self.workspace.set_full(index, value)
        elif isinstance(index, tuple):
            self.workspace.set(var_name=index[0], t=index[1], v=value)
        else:
            raise TypeError("index must be either str or tuple(str, int)".format())

    def get_by_name(self, n: str) -> list[Self]:
        """Returns the list of agents included in this agent that have a particular name."""
        if n == self._name:
            return [self]
        return []


class TimeAgent(Agent, ABC):
    """
    `TimeAgent` is used as a convention to represent agents that
    use a time index in their `__call__` function (not mandatory)
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def forward(self, t: int, **kwargs) -> Any:
        raise NotImplementedError(
            "Your TemporalAgent must override forward with a time index"
        )


class SerializableAgent(Agent, ABC):
    """
    `SerializableAgent` is used as a convention to represent agents that are serializable (not mandatory)
    You can override the serialize method to return the agent without the attributes that are not serializable.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def serialize(self) -> Self:
        """
        Return the `SerializableAgent` without the unsersializable attributes
        """
        try:
            return self
        except Exception as e:
            raise NotImplementedError(
                "Could not serialize your {c} SerializableAgent because of {e}\n"
                "You have to override the serialize method".format(
                    c=self.__class__.__name__, e=e
                )
            )

    def save(self, filename: str) -> None:
        """Save the agent to a file

        Args:
            filename (str): The filename to use
        """
        try:
            with open(filename, "wb") as f:
                pickle.dump(self.serialize(), f, pickle.DEFAULT_PROTOCOL)
        except Exception as e:
            raise Exception(
                "Could not save agent to file {filename} because of {e} \n"
                "Make sure to have properly overriden the serialize method.".format(
                    filename=filename, e=e
                )
            )


def load(filename: str) -> Agent:
    """Load the agent from a file

    Args:
        filename (str): The filename to use

    Returns:
        sabblium.Agent: The agent or a subclass of it
    """
    try:
        with open(filename, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise Exception(
            "Could not load agent from file {filename} because of {e}".format(
                filename=filename, e=e
            )
        )
