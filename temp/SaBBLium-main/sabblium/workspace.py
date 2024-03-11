#  SaBBLium ― A Python library for building and simulating multi-agent systems.
#
#  Copyright © Facebook, Inc. and its affiliates.
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

from __future__ import annotations

from typing import Generator, Iterable

import numpy as np
import torch
from torch import Tensor

from ._tensors import (
    CompactTemporalTensor,
    SlicedTemporalTensor,
    TemporalTensor,
)

""" This module provides different ways to store tensors that are more flexible than the torch.Tensor class
It also defines the `Workspace` as a dictionary of tensors.
"""


class Workspace:
    """
    Workspace is the most important class in `SaBBLium`.
    It corresponds to a collection of WorkspaceTensors.:

    In the most cases, we consider that all the tensors have the same time and batch sizes
    (but it is not mandatory for most of the functions)
    """

    def __init__(self, workspace: Workspace | None = None):
        """
        Create an empty workspace

        Args:
            workspace (Workspace, optional): If specified, it creates a copy of the workspace (where tensors are cloned as CompactTemporalTensors)
        """
        self.variables: dict[str, TemporalTensor] = {}
        if workspace is not None:
            for k in workspace.keys():
                self.set_full(k, workspace[k].clone())

    def __len__(self):
        return len(self.variables)

    @staticmethod
    def take_per_row_strided(
        tensor: Tensor, index: Tensor, num_elem: int = 2
    ) -> Tensor:
        # TODO: Optimize this function
        assert index.dtype in [torch.short, torch.int, torch.long]
        arange = torch.arange(tensor.size()[1], device=tensor.device)
        return torch.cat(
            [tensor[index + t, arange].unsqueeze(0) for t in range(num_elem)],
            dim=0,
        )

    def set(
        self,
        var_name: str,
        t: int,
        v: Tensor,
        batch_dims: tuple[int, int] | None = None,
    ):
        """Set the variable var_name at time t"""
        if var_name not in self.variables:
            self.variables[var_name] = SlicedTemporalTensor()
        elif isinstance(self.variables[var_name], TemporalTensor):
            self.variables[var_name] = self.variables[var_name].to_sliced()
        else:
            raise NotImplementedError(
                "Cannot set a variable at time {} that is not a TemporalTensor".format(
                    t
                )
            )
        self.variables[var_name].set(t=t, value=v, batch_dims=batch_dims)

    def get(
        self,
        var_name: str,
        t: int,
        batch_dims: tuple[int, int] | None = None,
    ) -> Tensor:
        """Get the variable var_name at time t"""
        assert var_name in self.variables, "Unknown variable '" + var_name + "'"
        return self.variables[var_name].get(t, batch_dims=batch_dims)

    def clear(self, name: str | None = None):
        """Remove all the variables from the workspace"""
        if name is None:
            for k, v in self.variables.items():
                v.clear()
            self.variables = {}
        else:
            self.variables[name].clear()
            del self.variables[name]

    def contiguous(self) -> Workspace:
        """Generates a workspace where all tensors are stored in the Compact format."""
        workspace = Workspace()
        for k in self.keys():
            workspace.set_full(var_name=k, value=self.get_full(k))
        return workspace

    def set_full(
        self,
        var_name: str,
        value: Tensor,
        batch_dims: tuple[int, int] | None = None,
    ):
        """
        Set variable var_name with a complete tensor of size (T × B × …)
        T is the time dimension and B is the batch size.
        """
        if var_name not in self.variables:
            self.variables[var_name] = CompactTemporalTensor()
        self.variables[var_name].set_full(value=value, batch_dims=batch_dims)

    def get_full(
        self, var_name: str, batch_dims: tuple[int, int] | None = None
    ) -> Tensor:
        """Return the complete tensor for var_name"""
        assert var_name in self.variables, (
            "[Workspace.get_full] unknown variable '" + var_name + "'"
        )
        return self.variables[var_name].get_full(batch_dims=batch_dims)

    def keys(self):
        """Return an iterator over the variables names"""
        return self.variables.keys()

    def __getitem__(self, key: str | Iterable[str]) -> Tensor | Generator[Tensor]:
        """
        If key is a string, then it returns a `torch.Tensor`.
        If key is a list or tuple of string, it returns a tuple of `torch.Tensor`.
        """
        if isinstance(key, str):
            return self.get_full(key)
        elif isinstance(key, Iterable):
            return (self.get_full(k) for k in key)
        else:
            raise ValueError("Invalid key type")

    def _all_variables_same_time_size(self) -> bool:
        """Check that all variables have the same time size"""
        _ts = None
        for k, v in self.variables.items():
            if _ts is None:
                _ts = v.time_size()
            if _ts != v.time_size():
                return False
        return True

    def time_size(self) -> int:
        """Return the time size of the variables in the workspace"""
        _ts = None
        for k, v in self.variables.items():
            if _ts is None:
                _ts = v.time_size()
            assert _ts == v.time_size(), "Variables must have the same time size"
        return _ts

    def batch_size(self) -> int:
        """Return the batch size of the variables in the workspace"""
        _bs = None
        for k, v in self.variables.items():
            if _bs is None:
                _bs = v.batch_size()
            assert _bs == v.batch_size(), "Variables must have the same batch size"
        return _bs

    def select_batch(self, batch_indexes: Tensor) -> Workspace:
        """Given a tensor of indexes, it returns a new workspace with the select elements (over the batch dimension)"""
        _bs = None
        for k, v in self.variables.items():
            if _bs is None:
                _bs = v.batch_size()
            assert _bs == v.batch_size(), "Variables must have the same batch size"

        workspace = Workspace()
        for k, v in self.variables.items():
            workspace.variables[k] = v.select_batch(batch_indexes)
        return workspace

    def select_batch_n(self, n: int):
        """Return a new Workspace of batch_size n by randomly sampling over the batch dimension"""
        who = torch.randint(low=0, high=self.batch_size(), size=(n,))
        return self.select_batch(who)

    def copy_time(
        self,
        from_time: int,
        to_time: int,
        n_steps: int,
        var_names: list[str] | None = None,
    ):
        """
        Copy all the variables values from time `from_time` to `from_time+n_steps` to `to_time` to `to_time+n_steps`.
        Eventually restricted to specific variables using `var_names`.
        """
        for k, v in self.variables.items():
            if var_names is None or k in var_names:
                v.copy_time(from_time, to_time, n_steps)

    def get_time_truncated(
        self,
        var_name: str,
        from_time: int,
        to_time: int,
        batch_dims: tuple[int, int] | None = None,
    ) -> Tensor:
        """Return workspace[var_name][from_time:to_time]"""
        assert 0 <= from_time < to_time and to_time >= 0

        v = self.variables[var_name]
        if isinstance(v, SlicedTemporalTensor):
            return v.get_time_truncated(from_time, to_time, batch_dims)
        else:
            return v.get_full(batch_dims)[from_time:to_time]

    def get_time_truncated_workspace(self, from_time: int, to_time: int) -> Workspace:
        """Return a workspace with all variables truncated between from_time and to_time"""
        workspace = Workspace()
        for k in self.keys():
            workspace.set_full(k, self.get_time_truncated(k, from_time, to_time))
        return workspace

    @staticmethod
    def cat_batch(workspaces: list[Workspace]) -> Workspace:
        """
        Concatenate multiple workspaces over the batch dimension.
        The workspaces must have the same time dimension.
        """

        ts = None
        for w in workspaces:
            if ts is None:
                ts = w.time_size()
            assert ts == w.time_size(), "Workspaces must have the same time_size"

        workspace = Workspace()
        for k in workspaces[0].keys():
            vals = [w[k] for w in workspaces]
            v = torch.cat(vals, dim=1)
            workspace.set_full(k, v)
        return workspace

    def copy_n_last_steps(self, n: int, var_names: list[str] | None = None):
        """Copy the n last timesteps of each variable to the n first timesteps."""
        _ts = None
        for k, v in self.variables.items():
            if var_names is None or k in var_names:
                if _ts is None:
                    _ts = v.time_size()
                assert _ts == v.time_size(), "Variables must have the same time size"

        for k, v in self.variables.items():
            if var_names is None or k in var_names:
                self.copy_time(_ts - n, 0, n)

    def zero_grad(self) -> None:
        """Remove any gradient information"""
        for k, v in self.variables.items():
            v.zero_grad()

    def to(self, device: torch.device) -> Workspace:
        """Return a workspace where all tensors are on a particular device"""
        workspace = Workspace()
        for k, v in self.variables.items():
            workspace.variables[k] = v.to(device)
        return workspace

    def subtime(self, from_t: int, to_t: int) -> Workspace:
        """
        Return a workspace restricted to a subset of the time dimension
        """
        assert (
            self._all_variables_same_time_size()
        ), "All variables must have the same time size"
        workspace = Workspace()
        for k, v in self.variables.items():
            workspace.variables[k] = v.subtime(from_t, to_t)
        return workspace

    def remove_variable(self, var_name: str):
        """Remove a variable from the Workspace"""
        del self.variables[var_name]

    def __str__(self):
        r = ["Workspace:"]
        for k, v in self.variables.items():
            r.append(
                "\t"
                + k
                + ": time_size = "
                + str(v.time_size())
                + ", batch_size = "
                + str(v.batch_size())
                + ", shape = "
                + str(tuple(v.tensor_size()))
            )
        return "\n".join(r)

    def select_subtime(self, t: torch.LongTensor, window_size: int) -> Workspace:
        """
        `t` is a tensor of size `batch_size` that provides one time index for each element of the workspace.
        Then the function returns a new workspace by aggregating `window_size` timesteps starting from index `t`
        This method allows to sample multiple windows in the Workspace.
        Note that the function may be quite slow.

        Args:
             t ([torch.Tensor]): a batch_size tensor of int time positions
             window_size ([type]): the output time size
        """

        assert t.dtype in [torch.short, torch.int, torch.long]
        _vars = {k: v.get_full(batch_dims=None) for k, v in self.variables.items()}
        workspace = Workspace()
        for k, v in _vars.items():
            workspace.set_full(
                var_name=k,
                value=self.take_per_row_strided(
                    tensor=v, index=t, num_elem=window_size
                ),
            )
        return workspace

    def sample_subworkspace(
        self, n_times: int, n_batch_elements: int, n_timesteps: int
    ) -> Workspace:
        """
        Sample a workspace from the  workspace. The process is the following:
                * Let us consider that workspace batch_size is B and time_size is T
                * For n_times iterations:
                    * We sample a time window of size n_timesteps
                    * We then sample a n_batch_elements elements on the batch size
                    * =>> we obtain a workspace of size n_batch_elements x n_timesteps
                * We concatenate all the workspaces collected (over the batch dimension)

        Args:
            n_times ([int]): The number of sub workspaces to sample (and concatenate)
            n_batch_elements ([int]): <=workspace.batch_size() : the number of batch elements to sample for each sub workspace
            n_timesteps ([int]): <=workspace.time_size() : the number of timesteps to keep

        Returns:
            [Workspace]: The resulting workspace
        """
        batch_size: int = self.batch_size()
        time_size: int = self.time_size()
        to_aggregate: list[Workspace] = []
        for _ in range(n_times):
            assert not n_timesteps > time_size
            mini_workspace: Workspace = self
            if n_timesteps < time_size:
                t = np.random.randint(time_size - n_timesteps)
                mini_workspace = self.subtime(t, t + n_timesteps)

            # Batch sampling
            if n_batch_elements < batch_size:
                idx_envs = torch.randperm(batch_size)[:n_batch_elements]
                mini_workspace = mini_workspace.select_batch(idx_envs)
            to_aggregate.append(mini_workspace)

        if len(to_aggregate) > 1:
            mini_workspace = Workspace.cat_batch(to_aggregate)
        else:
            mini_workspace = to_aggregate[0]
        return mini_workspace

    def get_transitions(self, filter_key: str) -> Workspace:
        """Return a new workspace containing the transitions of the current workspace.
            Each key of the current workspace have dimensions [n_step, n_env, key_dim]
            {
                Key1 :
                    [
                        [step1, step1, step1], # for env 1,2,3 ...
                        [step2, step2, step2], # for env 1,2,3 ...
                        ...
                    ]
                ...

            }

            Return a workspace of transitions with the following structure :
            Each key of the returned workspace have dimensions [2, n_transitions, key_dim]
            key[0][0], key[1][0] = (step_1, step_2) # for env 1
            key[0][1], key[1][1] = (step_1, step_2) # for env 2
            key[0][2], key[1][2] = (step_2, step_3) # for env 1
            key[0][3], key[1][3] = (step_2, step_3) # for env 2
            ...

            Filters every transitions [step_final, step_initial]

        Returns:
            [Workspace]: The resulting workspace of transitions
        """

        transitions = {}
        stopped = self[filter_key][:-1]
        for key in self.keys():
            array = self[key]

            # remove transitions (s_terminal -> s_initial)
            x = array[:-1][~stopped]
            x_next = array[1:][~stopped]
            transitions[key] = torch.stack([x, x_next])

        workspace = Workspace()
        for k, v in transitions.items():
            workspace.set_full(k, v)
        return workspace
