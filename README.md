# veery - a Validation Environment for Exposing RiskY models

# Installation
To run all capabilities, install dependencies via:

```
pip install -e ".[dev,experiments]"
```

# Defining an environment

To run veery methods on any problem, you first must create an environment (as a class) for that problem. The environment class must contain the following attributes:
-  `profile(time)` a function that returns a desired variable value when given a time. 
- 

The environment must also contain the following methods:
- reset()
- _get_observation()
- _log_history()
- step()
- get_snapshot()
- set_snapshot()
- rollout_from_snapshot()
