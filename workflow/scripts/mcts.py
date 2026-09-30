import numpy as np
import pandas as pd
from accessories import load_trained_model
import fuzzing
import env
import profiles

"""
The algorithms in this script are based on Algorithm 5.10 in Algorithm's for Validation by Kochenderfer et al.
"""

def robustness(p_actual, p_desired, p_threshold):
    """Calculates the robustness of a trajectory defined by the actual power and the desired power for all timesteps. This is based on signal temporal logic.

    Args:
        p_actual (array): Actual achieved power
        p_desired (array): Desired power at corresponding times to p_actual
        p_threshold (float): power specification as a percent deviation from profile
    """
    percent_diffs = abs(p_actual - p_desired)/p_desired
    rho = min(p_threshold - percent_diffs)
    return max(0, rho) # clip so that all failures get the same score, keeps the algorithm objective on most likely failures, not biggest failures


def lcb(node, c):
    Qs = [node.Q for node in node.children]
    Ns = [node.N for node in node.children]
    lcbs = []
    for Q, N in zip(Qs, Ns):
        lcbs.append(Q - c*np.sqrt(np.log(node.N)/N))
    
    best_child_index = np.argmin(np.array(lcbs))

    return node.children[best_child_index]


class MCTS:
    def __init__(self,  c, k, alpha, disturbance_dist, k_max, env, model):
        self.c = c # exploration constant
        self.k = k # progressive widening constant
        self.alpha = alpha # progressive widening exponent
        self.disturbance_dist = disturbance_dist # sampling returns x, a disturbance object
        self.k_max = k_max # number of iterations
        self.env = env
        self.model = model

    def initialize_tree(self, starting_state):
        return [Node(
            state=starting_state,
            N=1
        )]

    def score(self, rho, x, lam):
        """Scoring function to return Q for a node

        Args:
            rho (float): Robustness of a trajectory calculated via robustness()
            x (disturbance): Disturbance object
            lam (likelihood weight): how much weight to place on the likelihood of the disturbance versus the proximity to failure (captured by rho)

        Returns:
            float: Q value for a node
        """
        return rho - lam*self.disturbance_dist.logpdf(x)

class Node:
    def __init__(
            self, 
            state,
            parent = None,
            edge = None,
            children = None,
            N = 0,
            Q = 0
    ):
        self.state = state
        self.parent = parent
        self.edge = edge
        self.children = children
        self.N = N
        self.Q = Q

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
        if self.state["time"] >= self.state["runtime"]:
            return True
        else:
            return False

    def select(self, alg, tree):
        """Method to select a node

        Args:
            alg (MCTS object): MCTS algorithm parameters
            tree (list): list of nodes in the tree
        """
        c, k, alpha, node = alg.c, alg.k, alg.alpha, tree[0] # always start at the root
        while len(node.children) > k*(node.N**alpha):
            node = lcb(node, c)
        return node

    def extend(self, ):
        return 


    





if __name__ == "__main__":
    model = load_trained_model(load_dir="train_fivemillion")
    disturbance_dist = fuzzing.DisturbanceDistribution(
                            Do=fuzzing.Do(sigma_p=0.01, sigma_drum=0.005),
                            Da=fuzzing.Da(sigma_dtheta=0),
                            Ds=fuzzing.Ds(),
                        )
    training_kwargs, testing_kwargs = profiles.get_profile(name="test", max_failed_drums=0)
    alg = MCTS(
        c=1, 
        k=1, 
        alpha=1, 
        disturbance_dist=disturbance_dist, 
        k_max=10, 
        environment=env.HolosMulti(**testing_kwargs), 
        model= load_trained_model(load_dir="train_fivemillion")
        )
    