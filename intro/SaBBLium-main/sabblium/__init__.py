#  SaBBLium ― A Python library for building and simulating multi-agent systems.
#
#  Copyright © Facebook, Inc. and its affiliates.
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

from .agent import Agent, TimeAgent, SerializableAgent
from .agents.seeding import SeedableAgent
from .workspace import Workspace

trace_workspace = False
trace = []
trace_maximum_size = 10000


def instantiate_class(arguments):
    from importlib import import_module

    d = dict(arguments)
    classname = d["classname"]
    del d["classname"]
    module_path, class_name = classname.rsplit(".", 1)
    module = import_module(module_path)
    c = getattr(module, class_name)
    return c(**d)


def get_class(arguments):
    from importlib import import_module

    if isinstance(arguments, dict):
        classname = arguments["classname"]
        module_path, class_name = classname.rsplit(".", 1)
        module = import_module(module_path)
        c = getattr(module, class_name)
        return c
    else:
        classname = arguments.classname
        module_path, class_name = classname.rsplit(".", 1)
        module = import_module(module_path)
        c = getattr(module, class_name)
        return c


def get_arguments(arguments):
    d = dict(arguments)
    if "classname" in d:
        del d["classname"]
    return d
