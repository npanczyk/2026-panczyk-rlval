# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

**veery** (a **V**alidation **E**nvironment for **E**xposing **R**isk**Y** models) is a Python library for falsifying controllers, with a focus on RL controllers. It finds and ranks the disturbance sequences (sensor, agent, and environment) that make a controlled system break its specification. Most of the search is done with Monte Carlo Tree Search (MCTS) guided by signal-temporal-logic robustness, but the goal is to expand this to other methods.

- Research goal / question: Can we find rare failure modes for an AI-based controller for a nuclear reactor system?
- Target output (paper, thesis chapter, venue, deadline): Conference paper revisions (first draft already submitted)
- Primary case study: the Holos-Quad microreactor (`experiments/holos`), adapted from Tunkle et al. (2025), https://github.com/aims-umich/microdrum-marl
- Teaching / sanity-check case: the inverted pendulum (`experiments/inverted_pendulum`)

## Setup

```bash
conda env create -f environment.yml   # creates the `veery` env (Python 3.12)
conda activate veery
# or, in an existing env:
pip install -e ".[dev,experiments]"
```

- Core deps: numpy, pandas, scipy, matplotlib.
- `experiments` extra: stable-baselines3, tensorboard, gymnasium==1.3.0 (pinned on purpose: no reason, this could be changed).
- The base conda env does **not** have pytest. Run things inside `veery`.

## Repository layout

```
src/veery/                 # the installable library: keep it problem-agnostic
  fuzzing.py               # Disturbance, DisturbanceDistribution, ZeroDist
  mcts.py                  # MCTS, Node, run_MCTS, failure ranking/saving
  accessories.py           # check_spec (psi), robustness (rho), percent_diff
  viz.py                   # rollout, tree, and robustness plots
  direct_falsification.py  # stub, not implemented yet
experiments/               # problem-specific code; NOT part of the package
  holos/                   # gym env, point-kinetics model, profiles, SB3 training, rollouts, tutorial notebook
  inverted_pendulum/       # non-RL (proportional controller) env + tutorial notebook
tests/                     # pytest suite
```

The experiment modules import each other by bare name (`from pke import HolosPK`, `from helpers import scale`), so run them from inside their own experiment folder or with that folder on `sys.path`. `models/` and `results/` are gitignored output directories.

## Core abstractions

### Environment interface (what veery expects from a problem)
An environment used with `run_MCTS` must provide:
- `profile(time)`: the desired value at a given time
- `reset()`, `_get_observation(xo, ...)`, `_log_history(...)`
- `step(action, disturbance)`: takes an explicit `Disturbance` every step
- `get_snapshot()` / `set_snapshot(snap)`: the snapshot dict must contain `"time"`, `"runtime"`, and `"observation"` (used by `Node.is_terminal` and `Node.extend`)
- `rollout_from_snapshot(model, snap, disturbance_distribution)`: returns `(actual, desired)`
- `state_robustness(threshold, percent=True)`: per-step robustness margin. `percent` says whether `threshold` is a fractional or flat deviation from the profile; use `percent=False` when the profile can be 0 (e.g. the pendulum). Pass the same flag to `run_MCTS`, `check_spec`, and `plot_tree`.
- `history`: logged as a DataFrame with `Time`/ `Observed`/`Actual` / `Desired` columns, last two required for `check_spec`

`model` must have SB3-style `predict(observation, deterministic=True)` that returns `(action, state)`.


### Disturbances
- `Disturbance(xo, xa, xs)`: the observation (sensor), agent (action), and state (environment) disturbance for one step.
- `DisturbanceDistribution(Do, Da, Ds)`: each component is a problem-specific class with `sample(...)` (returns a dict, or an array for xa/xs) and `logpdf(...)`.
- If a sigma is 0, use `ZeroDist` instead of `norm(0, 0)` so that sampling and log-likelihood stay well-defined.
- Each experiment defines its own `Do`, `Da`, and `Ds` in `experiments/<name>/disturbances.py`.
- For nominal RL training, wrap `HolosMulti` in `Undisturbed` (or call `make_undisturbed_env`) so SB3 gets the standard `step(action)` signature.

### MCTS
- Hyperparameters: `c` (exploration), `k` and `alpha` (progressive widening), `lambda` (likelihood weight in the score), and `k_max` (iterations).
- Node score: mean rollout robustness minus `lambda` times the log-likelihood. MCTS **minimizes** the score through the lower confidence bound (LCB), so lower means closer to failure.
- The score currently counts only the **observation** disturbance in the log-likelihood, but this should be modified to track the action and state disturbances as well. The score is comprised of path robustness and log-likelihood of the disturbances.
- `run_MCTS` writes `tree.pkl` and the ranked failures to `results_path`.

## Testing

```bash
pytest                    # from repo root, in the veery env
pytest tests/test_fuzzing.py -k <name>
```

- This section needs to be re-done to only consider tests for veery.

## Common tasks

- Train a Holos controller: [`cd experiments/holos && python train.py`]
- Run MCTS falsification on Holos: [`cd experiments/holos && python holos_tutorial.ipynb`] 
- To do: write a hyperparameter tuning script for MCTS. Don't start this until Nataly prompts for it.
- To do: add tracker to quantify time to first failure in MCTS. Don't start this until Nataly prompts for it.
- To do: write an MCTS loop outside of the tutorial notebook. Don't start this until Nataly prompts for it.
- Run inverted pendulum tutorial: `experiments/inverted_pendulum/pendulum_tutorial.ipynb`
- Where trained models live / how to get them, use default trained model for now:  `models/default/best_model.zip`
- Compute resources (local vs. cluster, e.g. Great Lakes): mostly local. Flag runs that will be expensive and we will run them on a cluster.

## Code conventions

- Google-style docstrings (`Args:` / `Returns:`) on public functions and classes.
- Code is formatted with no particular style, but pep8 is preferred.
- Keep `src/veery` free of problem-specific code. Holos or pendulum logic belongs in `experiments/`.
- Inline comments explain the *why* of physics or algorithm choices. Keep that style.
- AI attribution: modules, classes, or tests that are largely AI-generated get a note in their docstring, e.g. *"generated by Claude … but modified and verified by Nataly"*. Add the same kind of note when generating substantial new code.
- Units and scaling: Holos power is fractional (0–1), drum angles are in degrees (0–180), and `helpers.scale` / `descale` convert to and from gym space. 

## Workflow

- Main branch: `main`. Work happens on feature branches (currently `structure-refactor`).
- Commit/PR expectations: nudge me when you think a commit is a good idea, but I will actually do the committing
- Things Claude should NOT do without asking: anything! Nataly should approve all changes so that she stays informed and up-to-date on the codebase.

## Domain notes / glossary

- **psi (ψ)**: a specification check on a trajectory that returns a bool (`check_spec`).
- **rho (ρ)**: STL robustness, the margin to violating the spec. A value below 0 means failure.
- **Profile**: the desired power trajectory over time (`experiments/holos/profiles.py`).

