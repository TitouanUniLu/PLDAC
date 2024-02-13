#  SaBBLium ― A Python library for building and simulating multi-agent systems.
#
#  Copyright © Facebook, Inc. and its affiliates.
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import torch
from torch import Tensor
from torch.utils import data

from sabblium import Agent, SeedableAgent


class ShuffledDatasetAgent(Agent, SeedableAgent):
    """An agent that read a dataset infinitely in a shuffle order."""

    def __init__(
        self,
        dataset: data.Dataset,
        batch_size: int,
        output_names: tuple[str, str] = ("x", "y"),
        **kwargs,
    ):
        """Create the agent
        Args:
            dataset ([torch.utils.data.Dataset]): the Dataset
            batch_size ([int]): The number of datapoints to write at each call
            output_names (tuple): The name of the variables. Default to ("x", "y").
        """
        super().__init__(**kwargs)
        self.output_names = output_names
        self.dataset: data.Dataset = dataset
        self.batch_size: int = batch_size
        self.ghost_params = torch.nn.Parameter(torch.randn(()))

    def forward(self, **kwargs):
        """Write a batch of data at timestep==0 in the workspace"""
        # TODO: use torch sampler
        vs = []
        for k in range(self.batch_size):
            idx = self.np_random.integers(len(self.dataset))
            x = self.dataset[idx]
            xs = []
            for xx in x:
                if isinstance(xx, Tensor):
                    xs.append(xx.unsqueeze(0))
                else:
                    xs.append(torch.tensor(xx).unsqueeze(0))
            vs.append(xs)

        vals = [torch.cat([v[k] for v in vs], dim=0) for k in range(len(vs[0]))]

        for name, value in zip(self.output_names, vals):
            self.set((name, 0), value.to(self.ghost_params.device))


class DataLoaderAgent(Agent):
    """An agent based on a DataLoader that read a single dataset
    Usage is: agent.forward(), then one has to check if agent.finished() is True or Not.
    If True, then no data have been written in the workspace since the reading of the dataset is terminated
    """

    def __init__(
        self,
        dataloader: data.dataloader,
        output_names: tuple[str, str] = ("x", "y"),
        **kwargs,
    ):
        """Create the agent based on a dataloader
        Args:
            dataloader ([DataLoader]): The underlying pytorch dataloader object
            output_names (tuple, optional): Names of the variable to write in the workspace. Defaults to ("x", "y").
        """
        super().__init__(**kwargs)
        self.dataloader: data.dataloader = dataloader
        self.iter = iter(self.dataloader)
        self.output_names: tuple[str, str] = output_names
        self._finished: bool = False
        self.ghost_params = torch.nn.Parameter(torch.randn(()))

    def reset(self):
        self.iter = iter(self.dataloader)
        self._finished = False

    def finished(self):
        return self._finished

    def forward(self, **kwargs):
        super().forward(**kwargs)
        try:
            output_values = next(self.iter)
        except StopIteration:
            self.iter = None
            self._finished = True
        else:
            for name, value in zip(self.output_names, output_values):
                self.set((name, 0), value.to(self.ghost_params.device))
