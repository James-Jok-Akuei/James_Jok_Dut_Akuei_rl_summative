// This file has been removed as per instructions.
# Video Script

Target length: `2 minutes to 2 minutes 30 seconds`  
Assignment requirement: `3 minutes max`

## Before Recording

- Turn camera on
- Share the entire screen
- Keep both the terminal and the simulation visible when running the final model
- Use this command for the demo:

```bash
.venv/bin/python main.py --registry-path results/final_demo_model.json --export-trace
```

## Script

### 0:00 - 0:20

Hello, my name is James Jok Dut Akuei. This project is an emergency medical drone delivery reinforcement learning system. The agent must pick up supplies, move safely through the environment, and deliver them to the patient.

### 0:20 - 0:45

The environment includes buildings, no-fly zones, limited battery, and a charging station. This makes the task more realistic because the agent must balance safety, movement, and energy use.

### 0:45 - 1:10

The action space includes hover, movement in eight directions, pickup, delivery, and recharge. The observation space includes positions, battery level, wind, and distances to obstacles and restricted zones.

### 1:10 - 1:35

The reward structure encourages successful and safe delivery. The agent is rewarded for pickup, progress, and delivery, and penalized for invalid actions, collisions, no-fly violations, battery loss, and timeout.

### 1:35 - 1:55

I trained four methods on the same environment: DQN, REINFORCE, PPO, and A2C. A2C performed best overall, so it was selected as the final demonstration model.

### 1:55 - 2:15

I will now run the best-performing model. In the terminal, you can see the action, reward, battery, and mission status. In the simulation window, you can see the drone moving toward the patient.

### 2:15 - 2:40

As the simulation runs, the agent picks up the package, follows a learned route, and delivers it to the patient. This shows that it learned the mission objective.

### 2:40 - 2:55

In conclusion, A2C was the strongest model because it was the most stable and reliable. This project shows how reinforcement learning can be used in a realistic mission-based delivery system.

## Short Backup Version

If you are running out of time, keep these points no matter what:

1. State the problem.
2. State the objective of the agent.
3. Briefly explain the reward structure.
4. Mention the four algorithms.
5. State that A2C performed best.
6. Run the simulation with terminal and GUI visible.
