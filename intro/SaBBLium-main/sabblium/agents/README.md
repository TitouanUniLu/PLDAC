# sabblium.agents

We propose a list of agents to reuse (see Documentation in the code)

## utils

* `Agents`: Execute multiple agents sequentially
* `TemporalAgent`: Execute one agent over multiple timesteps
* `CopyAgent`: An agent to create copies of variables
* `PrintAgent`: An agent that print variables

## envs

* `GymAgent`: An agent based on a gymnasium environment

## dataloader

* `ShuffledDatasetAgent`: An agent to read random batches in a torch.utils.data.Dataset
* `DataLoaderAgent`: An agent to do one pass over a complete dataset (based on a DataLoader)
