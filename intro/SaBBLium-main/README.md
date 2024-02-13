# `SaBBLium`: A Flexible and Simple Library for Learning Sequential Agents (including Reinforcement Learning)

## TL;DR :

`SaBBLium` is a lightweight library extending PyTorch modules for developing **sequential decision models**.
It can be used for **Reinforcement Learning** (including model-based with differentiable environments,
multi-agent RL, etc...), but also in a supervised/unsupervised learning settings
(for instance for NLP, Computer Vision, etc...).
It is derived from [`SaLinA`](https://github.com/facebookresearch/salina)  and [`BBRL`](https://github.com/osigaud/bbrl)
* It allows to write very complex sequential models (or policies) in few lines
* main difference with `BBRL` and `SaLinA` is that `SaBBLium` is compatible with `Gymnasium`:
  * No more `NoAutoResetGymAgent` or `AutoResetGymAgent` just a `GymAgent` that can be used in both cases depending on wether the `Gymnasium` environment contains an `AutoResetWrapper` or not.
  * You should now use `env/stopped` instead of `env/done` as a stop variable
* No multiprocessing / no remote agent or workspace yet
* You can easily save and load your models with `agent.save` and `Agent.load` by making them inherit from `SerializableAgent`, if they are not serializable, you have to override the `serialize` method.
* An ImageGymAgent has been added with adapted serialization
* Many typos have been fixed and type hints have been added

## Citing `SaBBLium`
`SaBBLium` being inspired from [`SaLinA`](https://github.com/facebookresearch/salina), please use this bibtex if you want to cite `SaBBLium` in your publications:


Please use this bibtex if you want to cite this repository in your publications:

Link to the paper: [SaLinA: Sequential Learning of Agents](https://arxiv.org/abs/2110.07910)

```
    @misc{salina,
        author = {Ludovic Denoyer, Alfredo de la Fuente, Song Duong, Jean-Baptiste Gaya, Pierre-Alexandre Kamienny, Daniel H. Thompson},
        title = {SaLinA: Sequential Learning of Agents},
        year = {2021},
        publisher = {Arxiv},salina_cl
        howpublished = {\url{https://gitHub.com/facebookresearch/salina}},
    }
```

## Quick Start

* Just clone the repo and
* with pip 21.3 or newer : `pip install -e .`

**For development, set up [pre-commit](https://pre-commit.com) hooks:**

* Run `pip install pre-commit`
    * or `conda install -c conda-forge pre-commit`
    * or `brew install pre-commit`
* In the top directory of the repo, run `pre-commit install` to set up the git hook scripts
* Now `pre-commit` will run automatically on `git commit`!
* Currently isort, black are used, in that order

## Organization of the repo

* [sabblium](sabblium) is the core library
  * [sabblium.agents](sabblium/agents) is the catalog of agents (the same as `torch.nn` but for agents)

## Dependencies

`SaBBLium` utilizes [`PyTorch`](https://github.com/pytorch/pytorch), [`Hydra`](https://github.com/facebookresearch/hydra) for configuring experiments, and [`Gymnasium`](https://github.com/Farama-Foundation/Gymnasium) for reinforcement learning environments.

## Note on the logger

We provide a simple Logger that logs in both [`TensorBoard`](https://github.com/tensorflow/tensorboard) format and [`wandb`](https://github.com/wandb/wandb), but also as pickle files that can be re-read to make tables and figures. See [logger](sabblium/logger.py). This logger can be easily replaced by any other logger.

## Description

**Sequential Decision Making is much more than Reinforcement Learning**

* Sequential Decision Making is about interactions:
 * Interaction with data (e.g. attention-models, decision tree, cascade models, active sensing, active learning, recommendation, etc….)
 * Interaction with an environment (e.g. games, control)
 * Interaction with humans (e.g. recommender systems, dialog systems, health systems, …)
 * Interaction with a model of the world (e.g. simulation)
 * Interaction between multiple entities (e.g. multi-agent RL)


### What `SaBBLium` is

* A sandbox for developing sequential models at scale.

* A small (300 hundred lines) 'core' code that defines everything you will use to implement `agents` involved in sequential decision learning systems.
  * It is easy to understand and use since it keeps the main principles of pytorch, just extending [`nn.Module`](https://pytorch.org/docs/stable/nn.html) to [`Agent`](/sabblium/agent.py) in order to handle the temporal dimension.
* A set of **agents** that can be combined (like pytorch modules) to obtain complex behaviors
* A set of references implementations and examples in different domains **Reinforcement Learning**, **Imitation Learning**, **Computer Vision**, with more to come...

### What `SaBBLium` is not

* Yet another reinforcement learning framework: `SaBBLium` is focused on **sequential decision-making in general**. It can be used for RL (which is our main current use-case), but also for supervised learning, attention models, multi-agent learning, planning, control, cascade models, recommender systems, among other use cases.
* A `library`: SaBBLium is just a small layer on top of pytorch that encourages good practices for implementing sequential models. Accordingly, it is very simple to understand and use, while very powerful.
* A `framework`: SaBBLium is not a framework, it is just a set of tools that can be used to implement any kind of sequential decision-making system.

## License

`SaBBLium` is released under the MIT license. See [LICENSE](LICENSE) for additional details about it.
