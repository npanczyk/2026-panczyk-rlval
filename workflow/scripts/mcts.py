import numpy as np
import pandas as pd
from accessories import load_trained_model
import fuzzing
import env
import profiles
from loops import rollout_from_snapshot

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
    percent_diffs = abs(np.array(p_actual) - np.array(p_desired))/np.array(p_desired)
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

    def initialize_tree(self, starting_state):
        return [Node(
            state=starting_state,
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

    def score(self, snapshot, x):
        """Scoring function to return Q for a node

        Args:
            snapshot (dict): Snapshot of environment state from env.get_snapshot()
            x (disturbance): Disturbance object

        Returns:
            float: Q value for a node
        """
        p_actual, p_desired = rollout_from_snapshot(self.env, self.model, snapshot, self.disturbance_dist)
        rho = robustness(p_actual, p_desired, self.p_threshold)
        return rho - self.lam*self.disturbance_dist.logpdf(x)

class Node:
    def __init__(
            self, 
            state,
            parent = None,
            edge = None,
            N = 0,
            Q = 0
    ):
        self.state = state
        self.parent = parent
        self.edge = edge
        self.children = []
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
        # score the node
        q = alg.score(
            snapshot = snapshot,
            x = x
        )
        child_node = Node(
            state = snapshot,
            parent = self,
            edge = (last_observation, action, x),
            N=1,
            Q=q
        )
        self.children.append(child_node)
        tree.append(child_node)

        # now it's time to backpropagate!
        # start at the parent node of the child we just created
        node = self
        # when we get to the root, node.parent = None and the loop ends
        while node is not None:
            node.N += 1
            node.Q += (q - node.Q) / node.N # Q values must be positive or this will break
            q, node = node.Q, node.parent
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
        lam = 0.01,
        disturbance_dist=disturbance_dist, 
        k_max=10, 
        env=env.HolosMulti(**testing_kwargs), 
        model= load_trained_model(load_dir="train_fivemillion")
        )
    tree = alg.initialize_tree()

    for i in range(alg.k_max):
        node = tree[0]
        node.select()
        node.expand()

    print(tree)

