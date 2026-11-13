# veery — a Validation Environment for Exposing RiskY models

veery is a Python library for **falsifying controllers**, especially reinforcement-learning (RL) controllers. Given a controlled system and a specification, it searches for disturbance sequences (sensor noise, actuator noise, and environment perturbations) that cause a system violate its specifications by some threshold value.

In addition to naive sampling, veery provides a targeted search for more likely failures using Monte Carlo Tree Search (MCTS).

The main case study is an RL controller for the Holos-Quad microreactor (adapted from [Tunkle et al., 2025](https://github.com/aims-umich/microdrum-marl)). We include a simpler inverted pendulum example for demonstration purposes. The falsification framework defined in this repo is based on that defined in *Algorithms for Validation* by [Kochenderfer et al.](https://algorithmsbook.com/validation/).

## Installation

```bash
conda env create -f environment.yml   # creates the `veery` env (Python 3.12)
conda activate veery
```

or, in an existing Python ≥3.12 environment:

```bash
pip install -e ".[dev,experiments]"
```

`dev` adds pytest; `experiments` adds stable-baselines3, tensorboard, and gymnasium for the example problems.

## Quick start

See `experiments/inverted_pendulum/pendulum_tutorial.ipynb` for a full walkthrough. In short:

```python
import numpy as np
from veery.fuzzing import DisturbanceDistribution
from veery.mcts import run_MCTS
from veery.viz import plot_tree, plot_tree_structure
import disturbances
from environment import InvertedPendulum, PendulumController

D = DisturbanceDistribution(
    Do=disturbances.Do(sigma_theta=0.1, sigma_omega=0.1),  # sensor
    Da=disturbances.Da(sigma_torque=0),                     # actuator
    Ds=disturbances.Ds(sigma_theta=0, sigma_omega=0),       # environment
)
pendulum = InvertedPendulum(m=1, l=1, dt=0.05, episode_length=1000, save_dir="results/demo")

tree = run_MCTS(
    environment=pendulum,
    model=PendulumController(),
    disturbance_dist=D,
    results_path="results/notebook",
    threshold=np.pi / 4,
    iterations=100,
    percent=False,   # flat threshold, since the desired angle is 0
)
```

## Defining a problem

veery is problem-agnostic. To use it, you supply an **environment**, a **model** (controller), and a **disturbance distribution**.

### Environment

| Member | Purpose |
|---|---|
| `profile(time)` | Desired value of the controlled variable at `time` |
| `state` | Current state, passed to `DisturbanceDistribution.sample()` |
| `reset()` | Reset to the initial condition |
| `step(action, disturbance)` | Advance one step under an explicit `Disturbance`; returns a gym-style `(obs, reward, terminated, truncated, info)` |
| `get_snapshot()` / `set_snapshot(snap)` | Save/restore full state. The snapshot dict must include `"time"`, `"runtime"`, and `"observation"` |
| `rollout_from_snapshot(model, snap, disturbance_distribution)` | Roll out to the end of the episode from `snap`; returns `(actual, desired)` lists |
| `state_robustness(threshold, percent=True)` | Robustness margin at the current step |

`percent=True` treats `threshold` as a fractional deviation from the profile. Use `percent=False` (a flat deviation) when the profile can be 0. Pass the same value to `run_MCTS`, `check_spec`, and `plot_tree`.

If you want to use `check_spec` or the rollout plots, also log a `history` DataFrame with `Actual` and `Desired` columns. Both examples organize this with `_get_observation(xo)` and `_log_history(...)` helpers.

### Model

Any object with an SB3-style `predict(observation, deterministic=True)` that returns `(action, state)`. A trained stable-baselines3 policy works directly. We demonstrate a simpler model via a proportional controller for the Inverted Pendulum system (see `PendulumController`).

### Disturbances

```
DisturbanceDistribution          → sample() returns Disturbance(xo, xa, xs)
 ├─ Do: ComponentDistribution    → observation (sensor) disturbances, xo = {name: value}
 ├─ Da: ComponentDistribution    → agent (action) disturbances, xa
 └─ Ds: ComponentDistribution    → state (environment) disturbances, xs
      └─ {name: DisturbedVariable(distribution, size)}
```

Define each component's disturbance distribution by subclassing `ComponentDistribution`:

```python
from veery.fuzzing import ComponentDistribution, DisturbedVariable, gaussian

class Do(ComponentDistribution):
    def __init__(self, sigma_p=0.01, sigma_drum=0.005):
        super().__init__({
            "p": DisturbedVariable(gaussian(sigma_p)),
            "drum_angles": DisturbedVariable(gaussian(sigma_drum), size=8),
        })
```

`gaussian(0)` returns a `ZeroDist`, so a zero-sigma component samples zeros and has a "dummy" log-likelihood value of 0. To check a distribution, call `D.plot_samples(D.Do, "p")`. If a disturbance depends on state or action, you can override `sample()` in your subclass to accept those conditional values.

## Running MCTS

`run_MCTS(environment, model, disturbance_dist, results_path, threshold, iterations=100, hyperparams=None, percent=True)`

| Hyperparameter | Meaning | Default |
|---|---|---|
| `c` | Exploration constant (lower confidence bound) | 0.5 |
| `k`, `alpha` | Progressive widening: a node expands while `children ≤ k·N^alpha` | 1, 0.25 |
| `lambda` | Weight on disturbance log-likelihood in the node score | 0.01 |

The node score is the mean rollout robustness minus λ·log-likelihood. MCTS minimizes it, so it favors trajectories that are both close to failure and likely.

**Outputs** written to `results_path`:
- `tree.pkl`: the full search tree (list of `Node`s)
- `failures.csv`: failures ranked by mean log-likelihood along the path
- `failure_sequences.pkl`: the disturbance and action sequence for each failure, plus the root state, for replay

**Plotting** (`veery.viz`): `plot_tree`, `plot_tree_structure`, `plot_rollouts`, `plot_overlapping_rollouts`, `plot_robustness`.

## Examples

- **`experiments/holos/`**: Holos-Quad microreactor. Gym environment (`HolosMulti`), point-kinetics model, power profiles, and SB3 training (`python train.py`). Demo in `holos_tutorial.ipynb`.
- **`experiments/inverted_pendulum/`**: inverted pendulum with a proportional controller (no RL). Demo in `pendulum_tutorial.ipynb`.

Experiments should be run from inside their own folder to properly manage imports.

## Repository layout

```
src/veery/
  fuzzing.py      # disturbance classes
  mcts.py         # MCTS, failure ranking and saving
  accessories.py  # check_spec (ψ), robustness (ρ)
  viz.py          # plotting
experiments/      # problem-specific code (not part of the package)
tests/            # pytest suite
```

## Testing

```bash
pytest
```


## License

MIT
