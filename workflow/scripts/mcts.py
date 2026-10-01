import numpy as np
import pandas as pd
import os
import pickle
from accessories import load_trained_model, robustness
import fuzzing
import env
import profiles
from loops import rollout_from_snapshot
from viz import plot_tree, plot_tree_structure
from tqdm import tqdm

"""
The functions in this script were inspired by Algorithm 5.10 in Algorithm's for Validation by Kochenderfer et al..
"""


def lcb(node, c):
    Qs = [node.Q for node in node.children]
    Ns = [node.N for node in node.children]
    lcbs = []
    for Q, N in zip(Qs, Ns):
        lcbs.append(Q - c*np.sqrt(np.log(node.N)/N))
    
    best_child_index = np.argmin(np.array(lcbs))

    return node.children[best_child_index]


class MCTS:
    def __init__(self,  c, k, alpha, lam, disturbance_dist, k_max, env, model, p_threshold=0.03):
        self.c = c # exploration constant
        self.k = k # progressive widening constant
        self.alpha = alpha # progressive widening exponent
        self.lam = lam # how much weight to place on the likelihood of the disturbance versus the proximity to failure (captured by rho) for the scoring function
        self.disturbance_dist = disturbance_dist # sampling returns x, a disturbance object
        self.k_max = k_max # number of iterations
        self.env = env
        self.model = model
        self.p_threshold = p_threshold # power deviation threshold percent diff, defaults to 3%

    def initialize_tree(self):
        # reset the environment
        self.env.reset()
        initial_state = self.env.get_snapshot()
        return [Node(
            state=initial_state,
            N=1
        )]

    def select(self, tree):
            """Method to select a node
    
            Args:
                alg (MCTS object): MCTS algorithm parameters
                tree (list): list of nodes in the tree
            """
            node = tree[0] # always start at the root
            while len(node.children) > self.k*(node.N**self.alpha):
                node = lcb(node, self.c)
            return node

    def score(self, snapshot, x, path_rho, N=3):
        """Scoring function to return Q for a node

        Args:
            snapshot (dict): Snapshot of environment state from env.get_snapshot()
            x (disturbance): Disturbance object
            N (int, optional): number of rollouts, defaults to 3

        Returns:
            float: Q value for a node
        """
        # check if the path_rho, including the node we're scoring has reached failure, if so, rho will get clipped to 0, so just set it and skip the rollout
        if path_rho < 0:
            return -self.lam * self.disturbance_dist.logpdf(x)

        rho_list = []
        for i in range(N):
            p_actual, p_desired = rollout_from_snapshot(self.env, self.model, snapshot, self.disturbance_dist)
            # we want the robustness to be the min robustness of the whole trajectory, including parent nodes, so check that
            rho_list.append(min(path_rho, robustness(p_actual, p_desired, self.p_threshold)))

        avg_rho = np.mean(np.array(rho_list))

        return avg_rho - self.lam*self.disturbance_dist.logpdf(x)

class Node:
    def __init__(
            self, 
            state,
            parent = None,
            edge = None,
            N = 0,
            Q = 0,
            path_rho = np.inf,
    ):
        self.state = state
        self.parent = parent
        self.edge = edge
        self.children = []
        self.N = N
        self.Q = Q
        self.path_rho = path_rho # minimum robustness along the path

        """ 
        INFO IN THE STATE SNAPSHOT DICT
        {
        "time": self.time,
        "runtime": self.runtime,
        "state": self.state.copy(),
        "_p": self._p,
        "_pnext": self._pnext,
        "_dp": self._dp,
        "_drum_angles": self._drum_angles.copy(),
        "masks": self.masks.copy(),
        "observation": {k: v.copy() for k, v in self._last_observation.items()},
        "last_history_row": list(self.history[-1]),
    }"""

    def is_terminal(self):
        if self.state["time"] >= self.state["runtime"] or self.path_rho < 0:
            return True
        else:
            return False

    def extend(self, alg, tree):
        # reset the environment to the node's state
        alg.env.set_snapshot(self.state)
        # get edge info
        last_observation = self.state["observation"]
        action, _states = alg.model.predict(last_observation, deterministic=True)
        # sample a disturbance distribution using the environment's state and action we just predicted
        x = alg.disturbance_dist.sample(alg.env.state, action)
        # now we actually create the node, by taking a step
        obs, _, terminated, truncated, _ = alg.env.step(
                    action=action,
                    disturbance = x
                )
        snapshot = alg.env.get_snapshot()
        # get the robustness margin for this step
        rho_step = robustness(p_actual=alg.env._p, 
                              p_desired= alg.env.profile(alg.env.time), 
                              p_threshold=alg.p_threshold)
        
        # update the child's path robustness based on this step
        child_path_rho = min(self.path_rho, rho_step)

        # score the node
        q = alg.score(
            snapshot = snapshot,
            x = x,
            path_rho = child_path_rho
        )
        child_node = Node(
            state = snapshot,
            parent = self,
            edge = (last_observation, action, x),
            N=1,
            Q=q,
            path_rho = child_path_rho
        )
        self.children.append(child_node)
        tree.append(child_node)
        # now it's time to backpropagate!
        self.backpropagate(q)
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
        total += disturbance_dist.logpdf(node.edge[2])
        depth += 1
        node = node.parent
    return total/depth if depth else 0

def rank_failures(tree, disturbance_dist):
    """Ranks the failure trajectories from a run.

    Args:
        tree (list): list of nodes after running a tree search
        disturbance_dist (disturbance distribution)

    Returns:
        list: sorted list of failed runs
    """
    fails = [n for n in tree
             if n.parent is not None and n.path_rho < 0 and n.parent.path_rho >= 0]
    return sorted(fails, key=lambda n: path_logp(n, disturbance_dist), reverse=True)

def extract_path(node):
    """Nodes from just below the root down to `node`, in order."""
    path = []
    while node.parent is not None:
        path.append(node)
        node = node.parent
    return path[::-1]

def save_failures(tree, disturbance_dist, results_dir):
    os.makedirs(results_dir, exist_ok=True)
    fails = rank_failures(tree, disturbance_dist)
    tree_idx = {id(n): i for i, n in enumerate(tree)}

    rows, sequences = [], {}
    for rank, n in enumerate(fails, start=1):
        path = extract_path(n)
        rows.append({
            "rank": rank,
            "mean_logpdf": path_logp(n, disturbance_dist),
            "depth": len(path),
            "fail_time": n.state["time"],
            "path_rho": n.path_rho,
            "tree_index": tree_idx[id(n)],
        })
        sequences[rank] = {
            "disturbances": [p.edge[2] for p in path],  # Disturbance objects
            "actions": [p.edge[1] for p in path],
        }

    pd.DataFrame(rows).to_csv(f"{results_dir}/failures.csv", index=False)
    with open(f"{results_dir}/failure_sequences.pkl", "wb") as f:
        pickle.dump({"root_state": tree[0].state, "sequences": sequences}, f)
    print(f"Saved {len(fails)} failures to {results_dir}")
    return rows

def run_MCTS(iterations=100, sigma_power=0.01, exploration=0.5):
    results_dir = f"../../results/MCTS_{iterations}_sigp{sigma_power}_c{exploration}"
    # make sure that path exists
    os.makedirs(results_dir, exist_ok=True)
    disturbance_dist = fuzzing.DisturbanceDistribution(
                            Do=fuzzing.Do(sigma_p=sigma_power, 
                                          sigma_drum=0.005),
                            Da=fuzzing.Da(sigma_dtheta=0),
                            Ds=fuzzing.Ds(),
                        )
    _, testing_kwargs = profiles.get_profile(name="test", max_failed_drums=0)
    alg = MCTS(
        c=exploration, # exploration constant
        k=0.5, # progressive widening constant
        alpha=0.1, # progressive widening exponent
        lam = 0.001, # likelihood weight for score
        disturbance_dist=disturbance_dist, 
        k_max=iterations, # max iterations
        env=env.HolosMulti(**testing_kwargs), 
        model= load_trained_model(load_dir="train_fivemillion")
        )
    tree = alg.initialize_tree()

    for i in tqdm(range(alg.k_max)):
        node = alg.select(tree)
        if node.is_terminal():
            # we need to backpropagate here so that N increases and the tree stops exploring terminal nodes
            node.backpropagate(node.Q)
        else:
            node.extend(alg, tree)

    # pickle the tree object
    try:
        with open(f'{results_dir}/tree.pkl', 'wb') as file:
            pickle.dump(tree, file)
    except RecursionError:
        print("Recursion Error: Could not save tree!")

    plot_tree(tree, profile=alg.env.profile, p_threshold=alg.p_threshold, save_path=f"{results_dir}/tree_{iterations}.png")
    plot_tree_structure(tree, save_path=f"{results_dir}/tree_structure_{iterations}.png")

    save_failures(tree, disturbance_dist, results_dir)


if __name__ == "__main__":
    run_MCTS(iterations=10000, sigma_power=0.01, exploration=0.08)

