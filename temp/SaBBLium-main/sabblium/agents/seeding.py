#  SaBBLium ― A Python library for building and simulating multi-agent systems.
#
#  Copyright © Facebook, Inc. and its affiliates.
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

from abc import ABC

from sabblium import Agent


class SeedableAgent(Agent, ABC):
    """
    `SeedableAgent` is used as a convention to represent agents that can be seeded.
    """

    def __init__(self, seed: int | None = None, **kwargs):
        super().__init__(**kwargs)
        self._seed = seed
