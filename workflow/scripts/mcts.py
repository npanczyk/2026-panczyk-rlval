import numpy as np
import pandas as pd
import pickle
from accessories import load_trained_model
import fuzzing
import env
import profiles
from loops import rollout_from_snapshot
from viz import plot_tree, plot_tree_structure

"""
The functions in this script were inspired by Algorithm 5.10 in Algorithm's for Validation by Kochenderfer et al..
"""

def robustness(p_actual, p_desired, p_threshold):
    """Calculates the robustness of a trajectory defined by the actual power and the desired power for all timesteps. This is based on signal temporal logic.

    Args:
        p_actual (array): Actual achieved power
        p_desired (array): Desired power at corresponding times to p_actual
        p_threshold (float): power specification as a percent deviation from profile
    """
    # if we're at a terminal node, we won't get power arrays, so return infinity (we're going to take a min of this rho and the path rho, so path rho will always get picked if we're at a terminal node)
    if type(p_actual) != list:
        p_actual = np.array([p_actual])
    if type(p_desired) != list:
        p_desired = np.array([p_desired])
    if len(p_actual) == 0:
        return np.inf
    percent_diffs = abs(np.array(p_actual) - np.array(p_desired))/np.array(p_desired)
    rho = min(p_threshold - percent_diffs)
    return rho


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

    def score(self, snapshot, x, path_rho):
        """Scoring function to return Q for a node

        Args:
            snapshot (dict): Snapshot of environment state from env.get_snapshot()
            x (disturbance): Disturbance object

        Returns:
            float: Q value for a node
        """
        # check if the path_rho, including the node we're scoring has reached failure, if so, rho will get clipped to 0, so just set it and skip the rollout
        if path_rho < 0:
            return -self.lam * self.disturbance_dist.logpdf(x)
        
        p_actual, p_desired = rollout_from_snapshot(self.env, self.model, snapshot, self.disturbance_dist)
        # we want the robustness to be the min robustness of the whole trajectory
        rho = min(path_rho, robustness(p_actual, p_desired, self.p_threshold))
        # clip the robustness at the failure threshold so that all failures look the same and we prioritize likelihood
        return max(rho , 0) - self.lam*self.disturbance_dist.logpdf(x)

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
            node.Q += (q - node.Q) / node.N # Q values must be positive or this will break
            q, node = node.Q, node.parent
        return 


# post-processing stuff
def path_logp(node, dist):
    total = 0.0
    while node.parent is not None:
        total += dist.logpdf(node.edge[2])
        node = node.parent
    return total

def ranked_failures(tree, dist):
    fails = [n for n in tree
             if n.parent is not None and n.path_rho < 0 and n.parent.path_rho >= 0]
    return sorted(fails, key=lambda n: path_logp(n, dist), reverse=True)

def run_MCTS(iterations=100):
    results_dir = "../../results/MCTS"
    disturbance_dist = fuzzing.DisturbanceDistribution(
                            Do=fuzzing.Do(sigma_p=0.01, sigma_drum=0.005),
                            Da=fuzzing.Da(sigma_dtheta=0),
                            Ds=fuzzing.Ds(),
                        )
    _, testing_kwargs = profiles.get_profile(name="test", max_failed_drums=0)
    alg = MCTS(
        c=1, # exploration constant
        k=1, # progressive widening constant
        alpha=0.5, # progressive widening exponent
        lam = 0.01, # likelihood weight for score
        disturbance_dist=disturbance_dist, 
        k_max=iterations, # max iterations
        env=env.HolosMulti(**testing_kwargs), 
        model= load_trained_model(load_dir="train_fivemillion")
        )
    tree = alg.initialize_tree()

    for i in range(alg.k_max):
        node = alg.select(tree)
        if node.is_terminal():
            # we need to backpropagate here so that N increases and the tree stops exploring terminal nodes
            node.backpropagate(node.Q)
        else:
            node.extend(alg, tree)

    # pickle the tree object
    with open(f'{results_dir}/tree.pkl', 'wb') as file:
        pickle.dump(tree, file)

    plot_tree(tree, profile=alg.env.profile, p_threshold=alg.p_threshold, save_path=f"{results_dir}/tree_{iterations}.png")
    plot_tree_structure(tree, save_path=f"{results_dir}/tree_structure_{iterations}.png")

if __name__ == "__main__":
    run_MCTS(iterations=100)

