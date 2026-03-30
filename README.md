# James_Jok_Dut_Akuei_rl_summative

## Emergency Medical Drone Delivery Reinforcement Learning Project

This repository contains a mission-based reinforcement learning project for emergency medical drone delivery. A custom Gymnasium environment was built to compare one value-based method and three policy-based methods on the same mission: DQN, REINFORCE, PPO, and A2C.

## Project Goal

The agent controls a drone that must pick up medical supplies, avoid obstacles and no-fly zones, manage battery usage, optionally recharge, and deliver the package to the patient before timeout. The environment introduces realistic mission constraints so the task goes beyond a simple toy grid world.

## Environment Summary

- Discrete action space with 12 actions: hover, movement, pickup, delivery, and recharge
- Observation vector includes location, battery, mission targets, hazard distances, wind, and action proximity flags
- Reward design encourages progress, successful pickup and delivery, energy awareness, and safe navigation

## Algorithms Compared

- DQN
- REINFORCE
- PPO
- A2C

## Best Result

The best-performing model was `A2C a2c_run_02`.

- Mean reward: `352.28`
- Success rate: `0.90`

Run the final demo with:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python main.py --registry-path results/final_demo_model.json --export-trace
```

## Repository Structure

```text
environment/
  custom_env.py
  rendering.py
training/
  dqn_training.py
  ppo_training.py
  a2c_training.py
  reinforce_training.py
  run_all_experiments.py
models/
  dqn/
  ppo/
  a2c/
  reinforce/
results/
plots/
utils/
main.py
requirements.txt
README.md
```

## Setup

Install dependencies:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Main Commands

Run the random environment visualization:

```bash
.venv/bin/python training/random_play.py --save-gif
```

Run all experiment sweeps:

```bash
.venv/bin/python training/run_all_experiments.py --device cpu
```

Generate comparison plots:

```bash
.venv/bin/python training/compare_algorithms.py
```

Run the best final model:

```bash
.venv/bin/python main.py --registry-path results/final_demo_model.json --export-trace
```

## Author

James Jok Dut Akuei
