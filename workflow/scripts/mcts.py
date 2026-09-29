import numpy as np
import pandas as pd
from accessories import load_trained_model

def Q(max_deviation):
    failure_threshold = 0.05
    return -1*max_deviation/failure_threshold

class MCTSNode:
    def __init__(self, N, Q, path_max, state_snapshot, last_obs, parent_node, edge, children, path):
        self.N = N
        self.Q = Q
        self.path_max
        return



if __name__ == "__main__":
    model = load_trained_model(load_dir="train_fivemillion")
    