#
#  Copyright © Sorbonne University.
#
#  This source code is licensed under the MIT license found in the
#  LICENSE file in the root directory of this source tree.
#

import numpy as np
import os
from typing import List

import hydra
import optuna
import yaml
from omegaconf import DictConfig

# %%
import torch
import torch.nn as nn

# %%
import gymnasium as gym

from bbrl.visu.plot_critics import plot_discrete_q, plot_critic

from bbrl.utils.functional import gae
from bbrl.utils.chrono import Chrono

# HYDRA_FULL_ERROR = 1
import matplotlib
import matplotlib.pyplot as plt

from tabularmazemdp.toolbox import sample_categorical
from tabularmazemdp.mdp import Mdp
from bbrl_gymnasium.envs.maze_mdp import MazeMDPEnv
from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder
from functools import partial

from scipy.interpolate import griddata

from scipy.stats import ttest_ind, mannwhitneyu, rankdata, median_test
import bootstrapped.bootstrap as bs
import bootstrapped.compare_functions as bs_compare
import bootstrapped.stats_functions as bs_stats

tests_list = [
    "t-test",
    "Welch t-test",
    "Mann-Whitney",
    "Ranked t-test",
    "bootstrap",
    "permutation",
]

matplotlib.use("TkAgg")


# %%
def get_trial_value(trial: optuna.Trial, cfg: DictConfig, variable_name: str):
    suggest_type = cfg["suggest_type"]
    args = cfg.keys() - ["suggest_type"]
    args_str = ", ".join([f"{arg}={cfg[arg]}" for arg in args])
    return eval(f'trial.suggest_{suggest_type}("{variable_name}", {args_str})')


def get_trial_config(trial: optuna.Trial, cfg: DictConfig):
    for variable_name in cfg.keys():
        if type(cfg[variable_name]) != DictConfig:
            continue
        else:
            if "suggest_type" in cfg[variable_name].keys():
                cfg[variable_name] = get_trial_value(
                    trial, cfg[variable_name], variable_name
                )
            else:
                cfg[variable_name] = get_trial_config(trial, cfg[variable_name])
    return cfg


def launch_optuna(cfg_raw, run_func):
    cfg_optuna = cfg_raw.optuna

    def objective(trial):
        cfg_sampled = get_trial_config(trial, cfg_raw.copy())

        try:
            trial_result: float = run_func(cfg_sampled, trial)
            return trial_result
        except optuna.exceptions.TrialPruned:
            return float("-inf")

    study = hydra.utils.call(cfg_optuna.study)
    study.optimize(func=objective, **cfg_optuna.optimize)

    file = open("best_params.yaml", "w")
    yaml.dump(study.best_params, file)
    file.close()


######


def actor_critic_averaging(cfg, trial=None) -> float:
    v = np.zeros(cfg.nb_repeats)
    if cfg.save_perf:
        fo = open(cfg.filename, "a")
    for i in range(cfg.nb_repeats):
        v[i] = actor_critic_v(cfg)
    perf = v.mean()
    data = [cfg.alpha_critic, cfg.alpha_actor, perf]
    if cfg.save_perf:
        np.savetxt(fo, data, fmt="%1.3f", newline=" ")
        fo.write("\n")
        fo.close()
    return perf


def evaluate(mdp, policy):
    x, _ = mdp.reset(uniform=True)
    done = False
    reward = 0

    while not done:
        # Perform a step of the MDP
        u = sample_categorical(policy[x])
        _, r, done, *_ = mdp.step(u)
        reward += r
    return reward


def renormalization(mdp, policy, x):
    total = sum(policy[x])
    for u1 in range(mdp.action_space.n):
        policy[x, u1] = policy[x, u1] / total


def actor_critic_v(cfg) -> float:
    mdp = gym.make("MazeMDP-v0", kwargs={"width": 5, "height": 5, "ratio": 0.2})
    nb_episodes = cfg.nb_episodes
    alpha_critic = cfg.alpha_critic
    alpha_actor = cfg.alpha_actor
    render = cfg.render

    if cfg.save_learning_curve:
        fo = open(cfg.filename, "a")
    learning_data = np.zeros(nb_episodes)

    v = np.zeros(mdp.nb_states)  # initial action values are set to 0
    policy = (
        np.ones((mdp.nb_states, mdp.action_space.n)) / mdp.action_space.n
    )  # action probabilities are uniform

    mdp.timeout = cfg.timeout  # max episode length

    if render:
        video_recorder = VideoRecorder(mdp, "videos/Actor-Critic.mp4", enabled=render)
        mdp.init_draw("Actor-Critic", recorder=video_recorder)

    for ep in range(nb_episodes):
        # Draw the first state of episode i using a uniform distribution over all states
        x, _ = mdp.reset(uniform=True)
        done = False

        while not done:
            if render:
                mdp.draw_v(v, recorder=video_recorder)

            # Perform a step of the MDP
            u = sample_categorical(policy[x])
            y, r, done, *_ = mdp.step(u)

            # Update the state-action value function with bootstrap
            delta = r + mdp.gamma * v[y] * (1 - done) - v[x]
            v[x] = v[x] + alpha_critic * delta
            # Update the policy
            # The probability of an action should never become negative nor null
            policy[x, u] = max(10e-8, policy[x, u] + alpha_actor * delta)

            renormalization(mdp, policy, x)
            # Update the agent position
            x = y
        # if cfg.save_learning_curve:
        # learning_data[ep] = np.linalg.norm(v[0])
        learning_data[ep] = np.linalg.norm(v)

    if render:
        mdp.draw_v(v, recorder=video_recorder)
        video_recorder.close()

    if cfg.save_learning_curve:
        np.savetxt(fo, learning_data, fmt="%1.3f", newline=" ")
        fo.write("\n")
        fo.close()

    return np.linalg.norm(v)


def eval_gridsearch(cfg):
    size = cfg.sample_size
    ech = np.around(np.linspace(0.1, 1, size), 2)
    vals = np.zeros((size, size))
    i = 0
    for alpha_critic in ech:
        j = 0
        # OmegaConf does not support float64
        cfg.alpha_critic = np.float64(alpha_critic).item()
        for alpha_actor in ech:
            # OmegaConf does not support float64
            cfg.alpha_actor = np.float64(alpha_actor).item()
            vals[i, j] = actor_critic_averaging(cfg)
            print(alpha_critic, alpha_actor, vals[i, j])
            j = j + 1
        i = i + 1


def find_max(data):
    x, y, vals = data[:, 0], data[:, 1], data[:, 2]
    idx = vals.argmax()
    print(
        f"index{idx}, alpha critic: {x[idx]}, alpha actor: {y[idx]}, best_val: {vals[idx]}"
    )
    return x[idx], y[idx]


###############


def plot_perf(data, name):
    plt.figure()
    plt.title("Landscape alpha_actor, alpha_critic")
    size = 30
    ech = np.around(np.linspace(0.1, 1, size), 2)

    x, y, vals = data[:, 0], data[:, 1], data[:, 2]

    X, Y = np.meshgrid(np.linspace(0.1, 1, size), np.linspace(0.1, 1, size))
    interpolated_vals = griddata((x, y), vals, (X, Y), method="linear")
    plt.imshow(interpolated_vals, cmap="RdYlGn_r")  # , aspect='auto')
    # plt.imshow(np.where(values > 0, values, -1), cmap="RdYlGn_r")
    plt.xlabel("alpha_critic")
    plt.ylabel("alpha_actor")
    plt.xticks(range(size), labels=ech, rotation=90)
    plt.yticks(range(size), labels=ech)
    plt.colorbar()

    plt.savefig(f"alphas_{name}.png")
    # plt.show()


###############


def optim_optuna(cfg):
    cfg.filename = cfg.init_filename + "_optuna.data"
    cfg.optuna.optimize.n_trials = cfg.sample_size * cfg.sample_size
    launch_optuna(cfg, actor_critic_averaging)


def optim_gridsearch(cfg):
    cfg.filename = cfg.init_filename + "_gridsearch.data"
    eval_gridsearch(cfg)


def find_best(cfg):
    data = np.loadtxt(cfg.filename)
    alpha_critic, alpha_actor = find_max(data)
    return alpha_critic, alpha_actor, data


###############


def run_permutation_test(all_data, n1, n2):
    np.random.shuffle(all_data)
    data_a = all_data[:n1]
    data_b = all_data[-n2:]
    return data_a.mean() - data_b.mean()


def compute_central_tendency_and_error(id_central, id_error, sample):

    try:
        id_error = int(id_error)
    except:
        pass

    if id_central == "mean":
        central = np.nanmean(sample, axis=1)
    elif id_central == "median":
        central = np.nanmedian(sample, axis=1)
    else:
        raise NotImplementedError

    if isinstance(id_error, int):
        low = np.nanpercentile(sample, q=int((100 - id_error) / 2), axis=1)
        high = np.nanpercentile(sample, q=int(100 - (100 - id_error) / 2), axis=1)
    elif id_error == "std":
        low = central - np.nanstd(sample, axis=1)
        high = central + np.nanstd(sample, axis=1)
    elif id_error == "sem":
        low = central - np.nanstd(sample, axis=1) / np.sqrt(sample.shape[0])
        high = central + np.nanstd(sample, axis=1) / np.sqrt(sample.shape[0])
    else:
        raise NotImplementedError

    return central, low, high


def run_test(test_id, data1, data2, alpha=0.05):
    """
    Compute tests comparing data1 and data2 with confidence level alpha
    :param test_id: (str) refers to what test should be used
    :param data1: (np.ndarray) sample 1
    :param data2: (np.ndarray) sample 2
    :param alpha: (float) confidence level of the test
    :return: (bool) if True, the null hypothesis is rejected
    """
    data1 = data1.squeeze()
    data2 = data2.squeeze()
    n1 = data1.size
    n2 = data2.size

    if test_id == "bootstrap":
        assert alpha < 1 and alpha > 0, "alpha should be between 0 and 1"
        res = bs.bootstrap_ab(
            data1,
            data2,
            bs_stats.mean,
            bs_compare.difference,
            alpha=alpha,
            num_iterations=1000,
        )
        rejection = np.sign(res.upper_bound) == np.sign(res.lower_bound)
        return rejection

    elif test_id == "t-test":
        _, p = ttest_ind(data1, data2, equal_var=True)
        return p < alpha

    elif test_id == "Welch t-test":
        _, p = ttest_ind(data1, data2, equal_var=False)
        return p < alpha

    elif test_id == "Mann-Whitney":
        _, p = mannwhitneyu(data1, data2, alternative="two-sided")
        return p < alpha

    elif test_id == "Ranked t-test":
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        ranks = rankdata(all_data)
        ranks1 = ranks[:n1]
        ranks2 = ranks[n1 : n1 + n2]
        assert ranks2.size == n2
        _, p = ttest_ind(ranks1, ranks2, equal_var=True)
        return p < alpha

    elif test_id == "permutation":
        all_data = np.concatenate([data1.copy(), data2.copy()], axis=0)
        delta = np.abs(data1.mean() - data2.mean())
        num_samples = 1000
        estimates = []
        for _ in range(num_samples):
            estimates.append(run_permutation_test(all_data.copy(), n1, n2))
        estimates = np.abs(np.array(estimates))
        diff_count = len(np.where(estimates <= delta)[0])
        return (1.0 - (float(diff_count) / float(num_samples))) < alpha

    else:
        raise NotImplementedError


def perform_test():
    optuna_perfs = np.loadtxt("./perfs_optuna.data")
    gridsearch_perfs = np.loadtxt("./perfs_gridsearch.data")

    optuna_perfs = optuna_perfs.transpose()
    gridsearch_perfs = gridsearch_perfs.transpose()
    nb_datapoints = optuna_perfs.shape[1]
    nb_steps = optuna_perfs.shape[0]

    legend = ["optuna", "gridsearch"]

    # what do you want to plot ?
    id_central = "median"  # 'mean'
    id_error = 80  # (percentiles), also: 'std', 'sem'

    # which test ?
    # possible : ['t-test', "Welch t-test", 'Mann-Whitney', 'Ranked t-test', 'bootstrap', 'permutation']
    test_id = "Welch t-test"  # recommended
    confidence_level = 0.01

    sample_size = 20
    sample1 = optuna_perfs[:, np.random.randint(0, nb_datapoints, sample_size)]
    sample2 = gridsearch_perfs[:, np.random.randint(0, nb_datapoints, sample_size)]

    # downsample for visualization purpose
    downsampling_fact = 5
    steps = np.arange(0, nb_steps, downsampling_fact)
    sample1 = sample1[steps, :]
    sample2 = sample2[steps, :]

    # test
    sign_diff = np.zeros([len(steps)])
    for i in range(len(steps)):
        sign_diff[i] = run_test(
            test_id, sample1[i, :], sample2[i, :], alpha=confidence_level
        )

    central1, low1, high1 = compute_central_tendency_and_error(
        id_central, id_error, sample1
    )
    central2, low2, high2 = compute_central_tendency_and_error(
        id_central, id_error, sample2
    )

    # plot
    _, ax = plt.subplots(1, 1, figsize=(20, 10))
    lab1 = plt.xlabel("training steps")
    lab2 = plt.ylabel("performance")

    plt.plot(steps, central1, linewidth=10)
    plt.plot(steps, central2, linewidth=10)
    plt.fill_between(steps, low1, high1, alpha=0.3)
    plt.fill_between(steps, low2, high2, alpha=0.3)
    leg = ax.legend(legend, frameon=False)

    # plot significative difference as dots
    idx = np.argwhere(sign_diff == 1)
    y = max(np.nanmax(high1), np.nanmax(high2))
    plt.scatter(steps[idx], y * 1.05 * np.ones([idx.size]), s=100, c="k", marker="o")

    # style
    for line in leg.get_lines():
        line.set_linewidth(10.0)
    ax.spines["top"].set_linewidth(5)
    ax.spines["right"].set_linewidth(5)
    ax.spines["bottom"].set_linewidth(5)
    ax.spines["left"].set_linewidth(5)

    plt.savefig(
        "./plot.png", bbox_extra_artists=(leg, lab1, lab2), bbox_inches="tight", dpi=100
    )
    plt.show()


@hydra.main(
    config_path="./configs/",
    # config_name="maze.yaml",
    config_name="maze_optuna.yaml",
)
def main(cfg: DictConfig):
    optim_optuna(cfg)
    alpha_critic_optuna, alpha_actor_optuna, vals = find_best(cfg)
    plot_perf(vals, "optuna")

    optim_gridsearch(cfg)
    alpha_critic_gridsearch, alpha_actor_gridsearch, vals = find_best(cfg)
    plot_perf(vals, "gridsearch")

    cfg.nb_repeats = 100
    cfg.save_learning_curve = True
    cfg.save_perf = False

    cfg.filename = cfg.init_filename + "s_optuna.data"
    cfg.alpha_critic = np.float64(alpha_critic_optuna).item()
    cfg.alpha_actor = np.float64(alpha_actor_optuna).item()
    actor_critic_averaging(cfg)

    cfg.filename = cfg.init_filename + "s_gridsearch.data"
    cfg.alpha_critic = np.float64(alpha_critic_gridsearch).item()  # 0.1 #
    cfg.alpha_actor = np.float64(alpha_actor_gridsearch).item()  # 0.1 #
    actor_critic_averaging(cfg)

    perform_test()


if __name__ == "__main__":
    main()

# Study the impact of the sample size on the significance of the test
