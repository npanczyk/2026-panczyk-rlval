import numpy as np
import pandas as pd
import os
import pickle
from veery.accessories import robustness
from veery.fuzzing import Disturbance, DisturbanceDistribution
from veery.viz import plot_tree, plot_tree_structure
from tqdm import tqdm
from pathlib import Path


def lcb(node, c):
    """Calculates the lower confidence bounds for all of a node's children and returns the node with the minimum LCB.

    Args:
        node (Node): Node from which a child is being chosen
        c (float): Exploration coefficient. Higher values preference exploration while lower values preference exploitation.

    Returns:
        Node: best child node to select based on LCB scores of all child nodes
    """
    live_children = []

    for child in node.children:
        if child.dead == False:
            live_children.append(child)
    # if all the children are dead, return None
    if not live_children:
        return None
    Qs = [child.Q for child in live_children]
    Ns = [child.N for child in live_children]
    lcbs = []
    for Q, N in zip(Qs, Ns):
        lcbs.append(Q - c * np.sqrt(np.log(node.N) / N))

    best_child_index = np.argmin(np.array(lcbs))

    return live_children[best_child_index]


class MCTS:
    def __init__(
        self,
        c,
        k,
        alpha,
        lam,
        max_children,
        disturbance_dist,
        k_max,
        environment,
        model,
        threshold,
        percent=True,
    ):
        self.c = c  # exploration constant
        self.k = k  # progressive widening constant
        self.alpha = alpha  # progressive widening exponent
        self.lam = lam  # how much weight to place on the likelihood of the disturbance versus the proximity to failure (captured by rho) for the scoring function
        self.max_children = max_children  # maximum number of children a node can have (important for avoiding getting stuck near a bunch of similar failures)
        self.disturbance_dist = (
            disturbance_dist  # sampling returns x, a disturbance object
        )
        self.k_max = k_max  # number of iterations
        self.environment = environment
        self.model = model
        self.threshold = threshold  # specification threshold
        self.percent = percent  # whether threshold is a fractional or flat deviation

    def initialize_tree(self):
        # reset the environment
        self.environment.reset()
        initial_state = self.environment.get_snapshot()
        return [Node(state=initial_state, N=1)]

    def select(self, tree):
        """Method to select a node

        Args:
            alg (MCTS object): MCTS algorithm parameters
            tree (list): list of nodes in the tree
        """
        node = tree[0]  # always start at the root
        # check progressive widening condition AND max children condition
        while (
            len(node.children) > self.k * (node.N**self.alpha)
            or len(node.children) >= self.max_children
        ):
            selected_child = lcb(node, self.c)
            # if all the children are dead, lcb will return None
            # if that happens, stop earlier than a leaf and expand
            if selected_child is None:
                break  # get out of the while loop, we're done searching
            node = selected_child
        return node

    def score(self, snapshot, x, path_rho, N=3):
        """Scoring function to return Q for a node

        Args:
            snapshot (dict): Snapshot of environment state from env.get_snapshot()
            x (Disturbance): Disturbance that led to node of interest
            N (int, optional): number of rollouts, defaults to 3

        Returns:
            float: Q value for a node
        """

        rho_list = []
        for i in range(N):
            actual, desired = self.environment.rollout_from_snapshot(
                model=self.model,
                snap=snapshot,
                disturbance_distribution=self.disturbance_dist,
            )
            # we want the robustness to be the min robustness of the whole trajectory, including parent nodes, so check for min between path and sampled rollout
            rho_list.append(
                min(
                    path_rho,
                    robustness(actual, desired, self.threshold, percent=self.percent),
                )
            )

        avg_rho = np.mean(np.array(rho_list))

        # get the loglikelihood of the observation disturbance only! not including action or state disturbances now
        obs_ll = self.disturbance_dist.logpdf(x)[0]
        return avg_rho - self.lam * obs_ll


class Node:
    def __init__(
        self,
        state,
        parent=None,
        edge=None,
        N=0,
        Q=0,
        path_rho=np.inf,
        dead=False,
    ):
        self.state = state
        self.parent = parent
        self.edge = edge
        self.children = []
        self.N = N
        self.Q = Q
        self.path_rho = path_rho  # minimum robustness along the path
        self.dead = dead  # dead means terminal or exhausted

    def __repr__(self):
        return (
            f"Node(t={self.state['time']}, N={self.N}, Q={self.Q:.3f}, "
            f"path_rho={self.path_rho:.3f}, children={len(self.children)})"
        )

    def is_terminal(self):
        if self.state["time"] >= self.state["runtime"] or self.path_rho < 0:
            self.dead = True
            return True
        else:
            return False

    def is_exhausted(self, max_children):
        if (
            np.all([c.dead == True for c in self.children])
            and len(self.children) == max_children
        ):
            self.dead = True
            return True
        else:
            return False

    def is_dead(self, max_children):
        if self.dead == True:
            return True
        elif self.is_terminal() or self.is_exhausted(max_children):
            self.dead = True
            return True
        else:
            return False

    def extend(self, alg, tree):
        # reset the environment to the node's state
        alg.environment.set_snapshot(self.state)
        # get edge info
        last_observation = self.state["observation"]
        action, _ = alg.model.predict(observation=last_observation, deterministic=True)
        # sample a disturbance distribution using the environment's state and action we just predicted
        x = alg.disturbance_dist.sample(state=alg.environment.state, action=action)
        # now we actually create the node, by taking a step
        obs, _, terminated, truncated, _ = alg.environment.step(
            action=action, disturbance=x
        )
        # take a snapshot of the environment at this updated state
        snapshot = alg.environment.get_snapshot()
        # get the robustness margin for this step
        rho_step = alg.environment.state_robustness(
            threshold=alg.threshold, percent=alg.percent
        )
        # update the child's path robustness based on this step
        child_path_rho = min(self.path_rho, rho_step)

        # score the node
        q = alg.score(snapshot=snapshot, x=x, path_rho=child_path_rho)
        child_node = Node(
            state=snapshot,
            parent=self,
            edge=(last_observation, action, x),
            N=1,
            Q=q,
            path_rho=child_path_rho,
        )
        self.children.append(child_node)
        tree.append(child_node)
        # now it's time to backpropagate!
        self.backpropagate(q)
        # after expanding, if the child node is terminal, check if the parent is dead, and propagate that up until you reach an alive node
        if child_node.is_terminal():
            node = self
            while node is not None and node.is_dead(alg.max_children):
                node = node.parent
        return

    def backpropagate(self, q):
        # start at the parent node of the child we just created
        # q is the child's score
        node = self
        # when we get to the root, node.parent = None and the loop ends
        while node is not None:
            node.N += 1
            node.Q += (q - node.Q) / node.N
            q, node = node.Q, node.parent
        return


# post-processing stuff
def path_logp(node, disturbance_dist):
    """Gets the loglikelihood of the disturbances along a full path (sum of individual loglikelihoods)

    Args:
        node (Node)
        disturbance_dist (disturbance distribution)

    Returns:
        float: loglikelihood of the disturbances along the whole path to the input node
    """
    total = 0.0
    depth = 0
    while node.parent is not None:
        # index 0th position to get logpdf of the observation only!
        total += disturbance_dist.logpdf(node.edge[2])[0]
        depth += 1
        node = node.parent
    return total / depth if depth else 0


def rank_failures(tree, disturbance_dist):
    """Ranks the failure trajectories from a run.

    Args:
        tree (list): list of nodes after running a tree search
        disturbance_dist (disturbance distribution)

    Returns:
        list: sorted list of failed runs
    """
    fails = [
        n
        for n in tree
        if n.parent is not None and n.path_rho < 0 and n.parent.path_rho >= 0
    ]
    return sorted(fails, key=lambda n: path_logp(n, disturbance_dist), reverse=True)


def extract_path(node):
    """Nodes from just below the root down to `node`, in order."""
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    return path[::-1]


def save_failures(tree, disturbance_dist, results_path):
    Path(results_path).mkdir(exist_ok=True)
    fails = rank_failures(tree, disturbance_dist)
    tree_idx = {id(n): i for i, n in enumerate(tree)}

    rows, sequences = [], {}
    for rank, n in enumerate(fails, start=1):
        path = extract_path(n)
        rows.append(
            {
                "rank": rank,
                "mean_logpdf": path_logp(n, disturbance_dist),
                "depth": len(path),
                "fail_time": n.state["time"],
                "path_rho": n.path_rho,
                "tree_index": tree_idx[id(n)],
            }
        )
        sequences[rank] = {
            "disturbances": [p.edge[2] for p in path],  # Disturbance objects
            "actions": [p.edge[1] for p in path],
        }

    pd.DataFrame(rows).to_csv(f"{results_path}/failures.csv", index=False)
    with open(f"{results_path}/failure_sequences.pkl", "wb") as f:
        pickle.dump({"root_state": tree[0].state, "sequences": sequences}, f)
    print(f"Saved {len(fails)} failures to {results_path}")
    return rows


def run_MCTS(
    environment,
    model,
    disturbance_dist,
    results_path,
    threshold,
    iterations=100,
    hyperparams=None,
    percent=True,
):
    """Runs Monte Carlo Tree Search for number of defined iterations.

    Args:
        environment (object): Problem environment that implements the veery interface (profile, state, reset, step, get_snapshot, set_snapshot, rollout_from_snapshot, state_robustness)
        model (object): Controller with an SB3-style predict(observation, deterministic=True) method that returns (action, state)
        disturbance_dist (DisturbanceDistribution): Distribution that disturbances are sampled from
        results_path (str): Folder where tree.pkl, failures.csv, and failure_sequences.pkl are saved
        threshold (float): Specification threshold, as a deviation from the profile
        iterations (int, optional): Number of MCTS iterations (k_max). Defaults to 100.
        hyperparams (dict, optional): MCTS hyperparameters with keys "c", "k", "alpha", and "lambda". Defaults to None, which uses c=0.5, k=1, alpha=0.25, lambda=0.01.
        percent (bool, optional): True if threshold is a fractional deviation from the profile, False if it is a flat deviation. Defaults to True.

    Returns:
        list: All nodes in the search tree, with the root at index 0
    """
    # check the path and make folders if necessary
    Path(results_path).mkdir(parents=True, exist_ok=True)

    # define the algorithm
    if hyperparams is None:
        hyperparams = {
            "c": 0.5,
            "k": 1,
            "alpha": 0.25,
            "lambda": 0.01,
            "max_children": 3,
        }

    alg = MCTS(
        c=hyperparams["c"],  # exploration constant
        k=hyperparams["k"],  # progressive widening constant
        alpha=hyperparams["alpha"],  # progressive widening exponent
        lam=hyperparams["lambda"],  # likelihood weight for score
        max_children=hyperparams["max_children"],
        disturbance_dist=disturbance_dist,
        k_max=iterations,  # max iterations
        environment=environment,
        model=model,
        threshold=threshold,  # error tolerance for determining failures
        percent=percent,
    )

    tree = alg.initialize_tree()

    for i in tqdm(range(alg.k_max)):
        if tree[0].is_dead(alg.max_children):
            print(f"Search space exhausted after {i} iterations!")
            break
        node = alg.select(tree)
        if node.is_terminal():
            # we need to backpropagate here so that N increases and the tree stops exploring terminal nodes
            node.backpropagate(node.Q)
        else:
            node.extend(alg, tree)

    # pickle the tree object
    try:
        with open(f"{results_path}/tree.pkl", "wb") as file:
            pickle.dump(tree, file)
    except RecursionError:
        print("Recursion Error: Could not save tree!")

    save_failures(tree, disturbance_dist, results_path)

    return tree
