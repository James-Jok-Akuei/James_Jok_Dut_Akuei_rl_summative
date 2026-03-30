// This file has been removed as per instructions.
# Reinforcement Learning Summative Assignment Report Outline

## Student Information

- Student Name: `[Your Name]`
- Video Recording: `[Insert your 3-minute video link here]`
- GitHub Repository: `[Insert your repository link here]`

## Project Overview

This project models an emergency medical drone that must collect life-saving supplies from a dispatch base and deliver them to a patient in a constrained urban airspace. The environment includes wind, battery management, recharge opportunities, buildings, and no-fly zones so that the agent solves a realistic operational mission instead of a toy navigation problem. Four reinforcement learning algorithms were compared on the same environment: DQN, REINFORCE, PPO, and A2C. Hyperparameter sweeps were used to study learning stability, convergence speed, exploration behavior, and generalization to unseen initial conditions. The objective was to identify which method best balances safety, efficiency, and mission completion in a logistics-style scenario that could integrate into a production dispatch pipeline.

## Environment Description

### Agent

The agent is a single autonomous medical drone. It can move in eight directions, hover, pick up a package, deliver to the patient, and recharge when near a valid station. Its behavior must balance mission speed with safety and energy management.

### Action Space

The action space is discrete with 12 actions:

1. Hover
2. Move North
3. Move South
4. Move East
5. Move West
6. Move North-East
7. Move North-West
8. Move South-East
9. Move South-West
10. Pick Up Supplies
11. Deliver Supplies
12. Recharge Battery

### Observation Space

The observation is a normalized numerical vector containing:

1. Drone coordinates
2. Dispatch base coordinates
3. Patient coordinates
4. Charging station coordinates
5. Battery percentage
6. Remaining step fraction
7. Payload and delivery flags
8. Direction to the current goal
9. Distance to the nearest obstacle
10. Distance to the nearest no-fly zone
11. Wind vector
12. Proximity flags for pickup, delivery, and recharge

### Reward Structure

Use the exact constants from `EnvironmentConfig` in [utils/config.py](/Users/apple/Documents/MLAssignmentsFolder/Mission-Based-Reinforcement-Learning/utils/config.py). A concise mathematical summary is:

- `R_t = step_penalty + progress_reward + interaction_reward + failure_penalty`
- Pickup gives a positive bonus
- Successful delivery gives the largest positive reward
- Moving closer to the active mission goal gives dense positive shaping
- Invalid actions, collisions, no-fly violations, battery depletion, and timeout give negative rewards

## System Analysis and Design

### DQN

Use a multilayer perceptron policy with replay memory, target-network updates, epsilon-greedy exploration, and batch updates from stored transitions.

### REINFORCE

Use a PyTorch multilayer perceptron policy trained with the vanilla Monte Carlo policy-gradient objective and entropy regularization for exploration.

### PPO

Use clipped policy updates, advantage estimation, and entropy regularization for stable policy optimization.

### A2C

Use synchronous actor-critic updates with entropy regularization and a learned value function baseline.

## Implementation Tables

Fill the four tables directly from:

- `results/dqn/sweep_summary.csv`
- `results/reinforce/sweep_summary.csv`
- `results/ppo/sweep_summary.csv`
- `results/a2c/sweep_summary.csv`

## Results Discussion

Insert and discuss:

1. `plots/overview/cumulative_rewards.png`
2. `plots/overview/training_stability.png`
3. `plots/overview/summary_bars.png`
4. Generalization numbers from `results/best_models.json` and each run summary JSON

## Conclusion and Discussion

Summarize which method performed best, which one was most stable, which one generalized best, and which hyperparameters most affected learning behavior in this environment.
