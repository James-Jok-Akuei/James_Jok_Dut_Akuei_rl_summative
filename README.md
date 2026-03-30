
# James_Jok_Dut_Akuei_rl_summative

# Emergency Medical Drone Delivery RL Assignment


This repository implements an Emergency Medical Drone Delivery mission for comparing value-based and policy-gradient reinforcement learning methods. The same custom Gymnasium environment is used by DQN, REINFORCE, PPO, and A2C, making the comparison consistent and reproducible.


## Mission Summary
The agent controls a drone that must collect emergency supplies from a dispatch base, navigate an urban airspace with obstacles and no-fly zones, manage battery usage, optionally recharge, and deliver the package to a patient before the mission times out. Wind is randomized per episode so the learned policy must adapt rather than memorize a fixed route.



## Project Structure (Required: James Jok Dut Akuei)

```
project_root/
├── environment/
│   ├── custom_env.py
│   └── rendering.py
├── training/
│   ├── dqn_training.py
│   └── pg_training.py
├── models/
│   ├── dqn/
│   └── pg/
├── main.py
├── requirements.txt
├── README.md
```

## Environment Design

`Action space`

- `0`: hover
- `1`: move north
- `2`: move south
- `3`: move east
- `4`: move west
- `5`: move northeast
- `6`: move northwest
- `7`: move southeast
- `8`: move southwest
- `9`: pick up supplies
- `10`: deliver supplies
- `11`: recharge battery

`Observation space`

The observation vector contains normalized agent coordinates, base/patient/charge coordinates, battery fraction, remaining step budget, payload flags, goal direction, nearest obstacle and no-fly distances, wind vector, and proximity flags for pickup, delivery, and recharge interactions.

`Reward highlights`

- Positive reward for successful pickup and successful delivery
- Dense progress reward for moving closer to the current mission goal
- Penalties for time usage, invalid actions, collisions, no-fly violations, and battery depletion
- Positive recharge reward when battery recovery is used meaningfully

## Training Commands

Install dependencies first:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Run the random-action visualization:

```bash
.venv/bin/python training/random_play.py --save-gif
```

Train the value-based model:

```bash
.venv/bin/python training/dqn_training.py --device cpu
```

Train the policy-gradient models:

```bash
.venv/bin/python training/ppo_training.py --device cpu
.venv/bin/python training/a2c_training.py --device cpu
.venv/bin/python training/reinforce_training.py --device cpu
```

Run every algorithm and rebuild the comparison outputs in one command:

```bash
.venv/bin/python training/run_all_experiments.py --device cpu
```

Optionally refine the strongest SB3 models for a final demo candidate without changing the completed sweep tables:

```bash
.venv/bin/python training/refine_best_sb3.py --algorithms a2c ppo --extra-timesteps 80000 --device cpu --eval-episodes 20
```

Generate comparison plots and the best-model registry:

```bash
.venv/bin/python training/compare_algorithms.py
```

Run the best-performing model in the GUI:

```bash
.venv/bin/python main.py --export-trace
```

Run the selected final demo agent from the refinement handoff:

```bash
.venv/bin/python main.py --registry-path results/final_demo_model.json --export-trace
```

Run quick smoke tests before launching the full 10-run sweeps:

```bash
.venv/bin/python training/dqn_training.py --device cpu --limit-runs 1 --timesteps-scale 0.02
.venv/bin/python training/pg_training.py --device cpu --algorithm all --limit-runs 1 --timesteps-scale 0.02 --episode-scale 0.1
```


## Notes

- All documentation and report files have been removed as per submission requirements.
- Only code, models, and required assets remain.
- The environment is implemented in `environment/custom_env.py`.
- Training scripts are in `training/`.
- Models are saved in `models/`.
- Plots and results are in their respective folders for reference.

For any further details, refer to the code and comments within each script.
