// This file has been removed as per instructions.
# Generated Report Summary

This file was generated from the completed experiment outputs and is intended to help fill the final PDF report.

## Best Performing Models

- `DQN`: `dqn_run_05` with mean reward `274.07` and success rate `0.00`.
- `REINFORCE`: `reinforce_run_08` with mean reward `89.84` and success rate `0.00`.
- `PPO`: `ppo_run_02` with mean reward `281.24` and success rate `0.60`.
- `A2C`: `a2c_run_02` with mean reward `352.28` and success rate `0.90`.

The final demo/deployment model is `A2C a2c_run_02` because it achieved the strongest held-out performance with mean reward `352.28` and success rate `0.90`.

## Results Discussion Draft

The cumulative reward plots show a clear separation between the strongest actor-critic methods and the weaker baselines in this environment. A2C achieved the best overall held-out performance, followed by PPO, while DQN obtained relatively high reward but failed to convert that reward into successful deliveries. This suggests that the value-based agent benefited from reward shaping but did not learn a reliable end-to-end delivery policy under the chosen configuration.

PPO and A2C displayed the most practical behavior for the emergency medical drone mission because they achieved non-zero generalization success on unseen states. A2C was the strongest model with a success rate of `0.90` and mean reward `352.28`, while PPO reached a success rate of `0.60` and mean reward `281.24`. REINFORCE learned more slowly and remained unstable, which is consistent with the higher variance normally associated with Monte Carlo policy-gradient methods.

The convergence metrics also support the qualitative plots. PPO converged earlier than A2C in episode count, but A2C achieved the highest final quality and the best final demo behavior. DQN converged according to reward trend, but because its success rate remained zero, convergence should be interpreted carefully: it converged to a suboptimal behavior rather than to the actual mission objective. This difference between reward optimization and task completion is an important finding for the report discussion.

## Figure References

- Cumulative rewards subplot: `/Users/apple/Documents/MLAssignmentsFolder/Mission-Based-Reinforcement-Learning/plots/overview/cumulative_rewards.png`
- Training stability subplot: `/Users/apple/Documents/MLAssignmentsFolder/Mission-Based-Reinforcement-Learning/plots/overview/training_stability.png`
- Convergence and generalization summary: `/Users/apple/Documents/MLAssignmentsFolder/Mission-Based-Reinforcement-Learning/plots/overview/summary_bars.png`
- Random-action visualization: `/Users/apple/Documents/MLAssignmentsFolder/Mission-Based-Reinforcement-Learning/results/random_policy_contact_sheet.png`

## DQN Table

| run_name | learning_rate | gamma | buffer_size | batch_size | exploration_fraction | mean_reward | success_rate | convergence_episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dqn_run_01 | 0.001 | 0.95 | 10000 | 64 | 0.2 | 86.23335476899175 | 0.0 | 1528 |
| dqn_run_02 | 0.00075 | 0.97 | 15000 | 64 | 0.22 | 109.23542385031973 | 0.0 | 1124 |
| dqn_run_03 | 0.0005 | 0.98 | 20000 | 128 | 0.15 | 82.71142151510223 | 0.0 | 1679 |
| dqn_run_04 | 0.0003 | 0.99 | 30000 | 128 | 0.1 | 59.29955207403945 | 0.0 | 959 |
| dqn_run_05 | 0.00025 | 0.985 | 25000 | 64 | 0.18 | 274.06524926757737 | 0.0 | 1659 |
| dqn_run_06 | 0.0002 | 0.99 | 35000 | 128 | 0.12 | 76.65633879886627 | 0.0 | 1274 |
| dqn_run_07 | 0.00015 | 0.995 | 40000 | 256 | 0.1 | 95.81156746508432 | 0.0 | 1017 |
| dqn_run_08 | 0.0001 | 0.97 | 18000 | 64 | 0.25 | 42.79316970010412 | 0.0 | 1191 |
| dqn_run_09 | 8e-05 | 0.99 | 50000 | 256 | 0.08 | 111.30663435018103 | 0.0 | 2133 |
| dqn_run_10 | 5e-05 | 0.995 | 60000 | 256 | 0.05 | 105.15137499681168 | 0.0 | 2157 |

## REINFORCE Table

| run_name | learning_rate | gamma | hidden_dim | entropy_coef | normalize_returns | episodes | mean_reward | success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reinforce_run_01 | 0.001 | 0.95 | 128 | 0.01 | True | 350 | -233.31250736094057 | 0.0 |
| reinforce_run_02 | 0.0007 | 0.97 | 128 | 0.015 | True | 400 | -121.0155474503644 | 0.0 |
| reinforce_run_03 | 0.0005 | 0.98 | 256 | 0.01 | True | 450 | 19.708818903129146 | 0.0 |
| reinforce_run_04 | 0.0003 | 0.99 | 256 | 0.02 | True | 500 | -165.4643733702798 | 0.0 |
| reinforce_run_05 | 0.0002 | 0.995 | 256 | 0.005 | True | 550 | -146.02630211076877 | 0.0 |
| reinforce_run_06 | 0.00015 | 0.97 | 128 | 0.03 | False | 400 | -97.78558090404022 | 0.0 |
| reinforce_run_07 | 0.0001 | 0.985 | 192 | 0.02 | False | 450 | -74.23669137477873 | 0.0 |
| reinforce_run_08 | 8e-05 | 0.99 | 256 | 0.01 | False | 500 | 89.83776952094108 | 0.0 |
| reinforce_run_09 | 5e-05 | 0.995 | 256 | 0.005 | False | 550 | -435.83335504055015 | 0.0 |
| reinforce_run_10 | 3e-05 | 0.997 | 256 | 0.001 | False | 600 | -166.2005001103637 | 0.0 |

## PPO Table

| run_name | learning_rate | gamma | n_steps | batch_size | gae_lambda | ent_coef | clip_range | mean_reward | success_rate | convergence_episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ppo_run_01 | 0.0003 | 0.99 | 512 | 64 | 0.95 | 0.01 | 0.2 | 137.94486474608422 | 0.0 | 719 |
| ppo_run_02 | 0.00025 | 0.98 | 512 | 128 | 0.95 | 0.015 | 0.2 | 281.2399902524058 | 0.6 | 971 |
| ppo_run_03 | 0.0002 | 0.99 | 1024 | 128 | 0.97 | 0.02 | 0.25 | 111.08629576159842 | 0.0 | 938 |
| ppo_run_04 | 0.00015 | 0.995 | 1024 | 256 | 0.98 | 0.01 | 0.15 | -82.62231853485136 | 0.0 | 1045 |
| ppo_run_05 | 0.0001 | 0.99 | 2048 | 256 | 0.95 | 0.005 | 0.2 | 69.45975875854467 | 0.0 | 1103 |
| ppo_run_06 | 0.0003 | 0.97 | 256 | 64 | 0.92 | 0.02 | 0.25 | 101.0920319795649 | 0.0 | 1005 |
| ppo_run_07 | 0.0002 | 0.985 | 512 | 64 | 0.9 | 0.03 | 0.2 | 232.50701297282814 | 0.4 | 1098 |
| ppo_run_08 | 0.00012 | 0.995 | 1024 | 128 | 0.99 | 0.008 | 0.18 | 68.99475875854466 | 0.0 | 1224 |
| ppo_run_09 | 8e-05 | 0.99 | 2048 | 256 | 0.95 | 0.002 | 0.12 | 64.59751794124914 | 0.0 | 717 |
| ppo_run_10 | 5e-05 | 0.995 | 2048 | 512 | 0.98 | 0.001 | 0.1 | -130.47520940303804 | 0.0 |  |

## A2C Table

| run_name | learning_rate | gamma | n_steps | ent_coef | vf_coef | mean_reward | success_rate | convergence_episode |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| a2c_run_01 | 0.0007 | 0.99 | 5 | 0.01 | 0.25 | 132.14413262383323 | 0.0 | 1066 |
| a2c_run_02 | 0.0005 | 0.98 | 10 | 0.015 | 0.3 | 352.28315213583465 | 0.9 | 1010 |
| a2c_run_03 | 0.0003 | 0.99 | 20 | 0.01 | 0.4 | 236.4369999999989 | 0.0 | 129 |
| a2c_run_04 | 0.00025 | 0.995 | 20 | 0.02 | 0.25 | 291.98204919379646 | 0.7 | 1345 |
| a2c_run_05 | 0.0002 | 0.99 | 40 | 0.005 | 0.5 | 88.71562018913707 | 0.0 | 1030 |
| a2c_run_06 | 0.00015 | 0.97 | 5 | 0.03 | 0.2 | 108.07907437419833 | 0.0 | 483 |
| a2c_run_07 | 0.00012 | 0.985 | 10 | 0.02 | 0.35 | 123.66694989573975 | 0.0 | 1020 |
| a2c_run_08 | 0.0001 | 0.99 | 30 | 0.01 | 0.45 | 69.45975875854467 | 0.0 | 917 |
| a2c_run_09 | 8e-05 | 0.995 | 50 | 0.005 | 0.55 | 69.45975875854467 | 0.0 | 1161 |
| a2c_run_10 | 5e-05 | 0.997 | 60 | 0.002 | 0.6 | 29.223476642038595 | 0.0 |  |

## Final Demo Command

```bash
.venv/bin/python main.py --registry-path results/final_demo_model.json --export-trace
```
