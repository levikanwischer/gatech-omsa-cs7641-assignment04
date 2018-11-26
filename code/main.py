# -*- coding: utf -*-

"""This module contains the main analysis methods."""

import csv
import importlib
import logging
import math
import os
import time
from copy import deepcopy
from itertools import product

import matplotlib

matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mdptoolbox import mdp

helpers = importlib.import_module("helpers")
envs = importlib.import_module("envs")
qlearning = importlib.import_module("qlearning")

matplotlib.rc("font", weight="normal", size=8)
sns.set_style("darkgrid")

np.random.seed(42)


@helpers.log_func_edges
class QLearning(qlearning.QLearning):
    """Modifying initialization of qlearning.QLearning."""

    def __init__(self, transitions, reward, discount, max_iter=1e6, **kwargs):
        params = {
            "transitions": transitions,
            "reward": reward,
            "discount": discount,
            "n_iter": max_iter,
            "interval": 25e4,
            "learning_rate": 0.1,  # best alpha from search
            "explore_interval": 100,  # best explore from search
        }

        params.update(kwargs)
        super().__init__(**params)


@helpers.log_func_edges
def _run_mdp_iteration(func, env, discount, iters=1e2, verbose=False):
    """Run MDP Iteration for a given environment."""
    env.reset()

    np.random.seed(42)

    P, R = envs.transitions_and_rewards(env)
    alg = func(P, R, discount, max_iter=iters)

    if verbose:
        alg.setVerbose()

    alg.run()
    return alg


@helpers.log_func_edges
def run_value_iteration(env, discount, iters=1e2, verbose=False):
    """Run Value Iteration for a given environment."""
    return _run_mdp_iteration(mdp.ValueIteration, env, discount, iters, verbose)


@helpers.log_func_edges
def run_policy_iteration(env, discount, iters=1e2, verbose=False):
    """Run Policy Iteration for a given environment."""
    return _run_mdp_iteration(mdp.PolicyIteration, env, discount, iters, verbose)


@helpers.log_func_edges
def run_q_learning(env, discount, iters=1e6, verbose=False):
    """Run Q Learning for a given environment."""
    return _run_mdp_iteration(QLearning, env, discount, iters, verbose)


@helpers.log_func_edges
def plot_value_and_policy_discounts(env, iters=1e2):
    """Run VI & PI for discounts & plots stats."""
    discounts = np.round([0.99] + np.linspace(0.90, 0.10, 9).tolist(), 2)
    valIter = {"time": list(), "iter": list()}
    polIter = {"time": list(), "iter": list()}

    np.random.seed(42)

    for discount in discounts:
        vi = run_value_iteration(env, discount, iters)
        pi = run_policy_iteration(env, discount, iters)

        valIter["time"].append(vi.time * 1000)
        polIter["time"].append(pi.time * 1000)

        valIter["iter"].append(vi.iter)
        polIter["iter"].append(pi.iter)

    plt.figure(figsize=(8, 4))
    for index, item in enumerate(valIter.keys()):

        plt.subplot(1, 2, index + 1)
        plt.title(f"{type(env).__name__}: Discount {item.title()}", fontsize=10)
        plt.xlabel("Discount Rates")
        plt.ylabel(f"{item.title()} Values")
        plt.grid()

        barWidth = 0.3

        plt.bar(
            np.arange(len(discounts)),
            valIter[item],
            width=barWidth,
            color="magenta",
            capsize=7,
            label="ValueIteration",
        )
        plt.bar(
            [x + barWidth for x in np.arange(len(discounts))],
            polIter[item],
            width=barWidth,
            color="cyan",
            capsize=7,
            label="PolicyIteration",
        )

        plt.xticks([r + barWidth for r in range(len(discounts))], discounts)

    plt.legend(loc="upper right")
    plt.tight_layout()

    outpath = os.path.join(helpers.IMGDIR, f"discount-iteration-{type(env).__name__}.png")
    plt.savefig(outpath)

    return discounts, valIter, polIter


@helpers.log_func_edges
def _plot_func_dimension_iterations(func, shapes, discount, iters=2.5e2):
    """Run VI & PI for dimensions & plots stats."""
    valIter = {"time": list(), "iter": list()}
    polIter = {"time": list(), "iter": list()}

    np.random.seed(42)

    sizes = list()
    for args in shapes:
        env = func(**args)
        vi = run_value_iteration(env, discount, iters)
        pi = run_policy_iteration(env, discount, iters)

        valIter["time"].append(vi.time * 1000)
        polIter["time"].append(pi.time * 1000)

        valIter["iter"].append(vi.iter)
        polIter["iter"].append(pi.iter)

        sizes.append(env.nS)

    plt.figure(figsize=(8, 4))
    for index, item in enumerate(valIter.keys()):

        plt.subplot(1, 2, index + 1)
        plt.title(f"{type(env).__name__}: Dimension {item.title()}", fontsize=10)
        plt.xlabel("Dimensions")
        plt.ylabel(f"{item.title()} Values")
        plt.grid()

        barWidth = 0.3

        plt.bar(
            np.arange(len(sizes)),
            valIter[item],
            width=barWidth,
            color="magenta",
            capsize=7,
            label="ValueIteration",
        )
        plt.bar(
            [x + barWidth for x in np.arange(len(sizes))],
            polIter[item],
            width=barWidth,
            color="cyan",
            capsize=7,
            label="PolicyIteration",
        )

        plt.xticks([r + barWidth for r in range(len(sizes))], sizes)

    plt.legend(loc="upper left")
    plt.tight_layout()

    outpath = os.path.join(helpers.IMGDIR, f"dimension-iteration-{type(env).__name__}.png")
    plt.savefig(outpath)

    return shapes, valIter, polIter


@helpers.log_func_edges
def plot_frozen_dimension_iterations(nMax=8, discount=0.90, iters=1e2):
    func = envs.MyFrozenLakeEnv
    shapes = [{"N": 4 * n, "M": 4 * n} for n in range(1, nMax + 1)]
    return _plot_func_dimension_iterations(func, shapes, discount, iters)


@helpers.log_func_edges
def plot_cliff_dimension_iterations(nMax=8, discount=0.90, iters=1e2):
    func = envs.MyCliffWalkingEnv
    shapes = [{"N": n, "clevels": n - 2} for n in range(3, nMax + 1)]
    return _plot_func_dimension_iterations(func, shapes, discount, iters)


@helpers.log_func_edges
def run_policy_episodes(env, policy, episodes, steps=1e2):
    """Run policy on problem (env) for n episodes."""
    timesteps, rewards, runtime = list(), list(), time.time()
    goal, hazard, other, alls = 0, 0, 0, 0

    np.random.seed(42)

    logging.info(f"Running {episodes} episodes...")

    for episode in range(episodes):
        observation = env.reset()

        done, _rewards, _steps = False, 0, steps
        while not done and _steps:

            _steps -= 1
            timestep = steps - _steps
            observation, reward, done, _ = env.step(policy[observation])

            _rewards += reward

        timesteps.append(timestep)
        rewards.append(_rewards)

        position = np.unravel_index(env.s, env.shape)
        fstate = "H" if env._hazard[position] else "o"
        fstate = "T" if position == env.end_state_tuple else fstate

        logging.info(
            f"{episode} Timestep == {timestep} & Rewards == {_rewards} & State == {fstate}"
        )

        goal += 1 if fstate == "T" else 0
        hazard += 1 if fstate in ("H", "C") else 0
        other += 1 if fstate == "o" else 0
        alls += 1

        if alls != (goal + hazard + other):
            logging.warning(f"Counter {alls} does not equal G/H/O {goal}/{hazard}/{other}.")

    runtime = time.time() - runtime
    logging.info(f"Total Steps {alls}, Goals {goal}, Hazards {hazard}, & Others {other}")

    return timesteps, rewards, (goal, hazard, other, alls), runtime


@helpers.log_func_edges
def create_policy_episodes_table(env, discount, episodes):
    """Create table of aggregate policy stats from n episodes."""
    algorithms = [
        run_value_iteration(env, discount),
        run_policy_iteration(env, discount),
        run_q_learning(env, discount),
    ]

    results = list()
    for algorithm in algorithms:
        steps, rewards, stats, runetime = run_policy_episodes(env, algorithm.policy, episodes)

        results.append(
            {
                "Algorithm": type(algorithm).__name__,
                "Learning": np.round(algorithm.time * 1000, 2),
                "Goals": np.round(stats[0], 2),
                "Steps": np.round(np.mean(steps), 2),
                "Rewards": np.round(np.mean(rewards), 2),
                "Match VI": algorithm.policy == algorithms[0].policy,
                "Match PI": algorithm.policy == algorithms[1].policy,
                "Match QL": algorithm.policy == algorithms[2].policy,
            }
        )

    outpath = os.path.join(helpers.IMGDIR, f"policy-episodes-{type(env).__name__}.csv")
    with open(outpath, "w", newline="\n") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=list(results[0].keys()))

        writer.writeheader()
        writer.writerows(results)

    return results


# FIXME: This sucks, and is incorrect, not sure how else to accumulate iterations though
@helpers.log_func_edges
def plot_q_learning_rates(env, discount, iters=1e6, episodes=20):
    """Plot reward, time, & steps results for q learning."""
    assert iters >= 1e4, "Iters must be greater than 1e4"
    jumps = [i // 1 for i in np.linspace(1e4, iters, iters // 1e4)]

    np.random.seed(42)

    results = {"Steps": list(), "Rewards": list(), "Time": list()}
    for jump in jumps:
        logging.info(f"Running Q Rates for {jump} iterations for {type(env).__name__}.")

        env.reset()

        P, R = envs.transitions_and_rewards(env)
        ql = QLearning(P, R, discount, max_iter=jump)
        ql.run()

        steps, rewards, _, runtime = run_policy_episodes(env, ql.policy, episodes)

        results["Steps"].append(np.round(np.mean(steps), 2))
        results["Rewards"].append(np.round(np.mean(rewards), 2))
        results["Time"].append(np.round(runtime * 1000 / episodes, 2))

    plt.figure(figsize=(12, 4))
    for index, item in enumerate(results.keys()):

        plt.subplot(1, 3, index + 1)
        plt.title(f"{type(env).__name__}: QLearning {item.title()}", fontsize=10)
        plt.xlabel("Iterations")
        plt.ylabel(f"{item.title()}")
        plt.grid()

        _pkwargs = {"linewidth": 1, "markersize": 2}
        plt.plot(jumps, results[item], marker="o", color="magenta", **_pkwargs)

    plt.tight_layout()

    outpath = os.path.join(helpers.IMGDIR, f"qlearning-iteration-{type(env).__name__}.png")
    plt.savefig(outpath)

    return jumps, results


def plot_best_q_learning_params(env, discount=0.9, iters=1e6, episodes=100, **kwargs):
    """Plot stats for various q learning params."""
    P, R = envs.transitions_and_rewards(env)

    alphas = [0.1, 0.3, 0.5, 0.7, 0.9]
    explores = [50, 100, 250]

    results, params = {"Steps": list(), "Rewards": list()}, list()
    for alpha, explore in product(alphas, explores):
        logging.info(f"{type(env).__name__} QL for Alpha={alpha} Explore={explore}")

        ql = QLearning(P, R, discount, iters, learning_rate=alpha, explore_interval=explore)
        ql.run()

        envs.render_and_policy(env, ql.policy)
        steps, rewards, *_ = run_policy_episodes(env, ql.policy, episodes, 250)

        results["Steps"].append(np.round(np.mean(steps), 2))
        results["Rewards"].append(np.round(np.mean(rewards), 2))

        params.append((alpha, explore))

    plt.figure(figsize=(8, 4))
    for index, item in enumerate(results.keys()):

        plt.subplot(1, 2, index + 1)
        plt.title(f"{type(env).__name__}: {type(ql).__name__} {item.title()}", fontsize=10)
        plt.xlabel("(alpha, explore)")
        plt.ylabel(f"{item.title()}")
        plt.grid()

        plt.bar(np.arange(len(params)), results[item], color="magenta", capsize=7, label=item)

        plt.xticks(np.arange(len(params)), params, rotation="vertical")

    plt.tight_layout()

    _cname = type(env).__name__ if "name" not in kwargs else kwargs["name"]
    outpath = os.path.join(helpers.IMGDIR, f"qlearning-params-{_cname}.png")
    plt.savefig(outpath)

    return params, results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(levelname)s - %(module)s:L%(lineno)s - %(message)s",
    )

    small = envs.MyFrozenLakeEnv()
    large = envs.MyCliffWalkingEnv()

    plot_value_and_policy_discounts(small)
    plot_value_and_policy_discounts(large)

    plot_best_q_learning_params(small)
    plot_best_q_learning_params(large)

    custom = envs.MyFrozenLakeEnv(16, 16)
    plot_best_q_learning_params(custom, name="MyFrozenLakeEnv16x16")

    create_policy_episodes_table(small, 0.90, 10000)
    create_policy_episodes_table(large, 0.90, 10000)

    plot_frozen_dimension_iterations(7, 0.90)
    plot_cliff_dimension_iterations(12, 0.90)

    for env in (small, large):
        env.reset()
        vi = run_value_iteration(env, 0.9, iters=1e2, verbose=False)
        envs.render_and_policy(env, vi.policy)
