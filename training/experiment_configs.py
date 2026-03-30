from __future__ import annotations


DQN_SWEEP = [
    {"run_name": "dqn_run_01", "learning_rate": 1e-3, "gamma": 0.95, "buffer_size": 10000, "batch_size": 64, "exploration_fraction": 0.20, "target_update_interval": 250, "train_freq": 4, "gradient_steps": 1, "total_timesteps": 30000},
    {"run_name": "dqn_run_02", "learning_rate": 7.5e-4, "gamma": 0.97, "buffer_size": 15000, "batch_size": 64, "exploration_fraction": 0.22, "target_update_interval": 300, "train_freq": 4, "gradient_steps": 1, "total_timesteps": 32000},
    {"run_name": "dqn_run_03", "learning_rate": 5e-4, "gamma": 0.98, "buffer_size": 20000, "batch_size": 128, "exploration_fraction": 0.15, "target_update_interval": 400, "train_freq": 4, "gradient_steps": 2, "total_timesteps": 36000},
    {"run_name": "dqn_run_04", "learning_rate": 3e-4, "gamma": 0.99, "buffer_size": 30000, "batch_size": 128, "exploration_fraction": 0.10, "target_update_interval": 500, "train_freq": 8, "gradient_steps": 2, "total_timesteps": 40000},
    {"run_name": "dqn_run_05", "learning_rate": 2.5e-4, "gamma": 0.985, "buffer_size": 25000, "batch_size": 64, "exploration_fraction": 0.18, "target_update_interval": 350, "train_freq": 4, "gradient_steps": 1, "total_timesteps": 36000},
    {"run_name": "dqn_run_06", "learning_rate": 2e-4, "gamma": 0.99, "buffer_size": 35000, "batch_size": 128, "exploration_fraction": 0.12, "target_update_interval": 600, "train_freq": 8, "gradient_steps": 4, "total_timesteps": 42000},
    {"run_name": "dqn_run_07", "learning_rate": 1.5e-4, "gamma": 0.995, "buffer_size": 40000, "batch_size": 256, "exploration_fraction": 0.10, "target_update_interval": 800, "train_freq": 8, "gradient_steps": 4, "total_timesteps": 45000},
    {"run_name": "dqn_run_08", "learning_rate": 1e-4, "gamma": 0.97, "buffer_size": 18000, "batch_size": 64, "exploration_fraction": 0.25, "target_update_interval": 300, "train_freq": 4, "gradient_steps": 1, "total_timesteps": 34000},
    {"run_name": "dqn_run_09", "learning_rate": 8e-5, "gamma": 0.99, "buffer_size": 50000, "batch_size": 256, "exploration_fraction": 0.08, "target_update_interval": 1000, "train_freq": 16, "gradient_steps": 8, "total_timesteps": 50000},
    {"run_name": "dqn_run_10", "learning_rate": 5e-5, "gamma": 0.995, "buffer_size": 60000, "batch_size": 256, "exploration_fraction": 0.05, "target_update_interval": 1200, "train_freq": 16, "gradient_steps": 8, "total_timesteps": 52000},
]

PPO_SWEEP = [
    {"run_name": "ppo_run_01", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 512, "batch_size": 64, "gae_lambda": 0.95, "ent_coef": 0.01, "clip_range": 0.2, "total_timesteps": 40000},
    {"run_name": "ppo_run_02", "learning_rate": 2.5e-4, "gamma": 0.98, "n_steps": 512, "batch_size": 128, "gae_lambda": 0.95, "ent_coef": 0.015, "clip_range": 0.2, "total_timesteps": 42000},
    {"run_name": "ppo_run_03", "learning_rate": 2e-4, "gamma": 0.99, "n_steps": 1024, "batch_size": 128, "gae_lambda": 0.97, "ent_coef": 0.02, "clip_range": 0.25, "total_timesteps": 45000},
    {"run_name": "ppo_run_04", "learning_rate": 1.5e-4, "gamma": 0.995, "n_steps": 1024, "batch_size": 256, "gae_lambda": 0.98, "ent_coef": 0.01, "clip_range": 0.15, "total_timesteps": 48000},
    {"run_name": "ppo_run_05", "learning_rate": 1e-4, "gamma": 0.99, "n_steps": 2048, "batch_size": 256, "gae_lambda": 0.95, "ent_coef": 0.005, "clip_range": 0.2, "total_timesteps": 50000},
    {"run_name": "ppo_run_06", "learning_rate": 3e-4, "gamma": 0.97, "n_steps": 256, "batch_size": 64, "gae_lambda": 0.92, "ent_coef": 0.02, "clip_range": 0.25, "total_timesteps": 38000},
    {"run_name": "ppo_run_07", "learning_rate": 2e-4, "gamma": 0.985, "n_steps": 512, "batch_size": 64, "gae_lambda": 0.90, "ent_coef": 0.03, "clip_range": 0.2, "total_timesteps": 40000},
    {"run_name": "ppo_run_08", "learning_rate": 1.2e-4, "gamma": 0.995, "n_steps": 1024, "batch_size": 128, "gae_lambda": 0.99, "ent_coef": 0.008, "clip_range": 0.18, "total_timesteps": 50000},
    {"run_name": "ppo_run_09", "learning_rate": 8e-5, "gamma": 0.99, "n_steps": 2048, "batch_size": 256, "gae_lambda": 0.95, "ent_coef": 0.002, "clip_range": 0.12, "total_timesteps": 55000},
    {"run_name": "ppo_run_10", "learning_rate": 5e-5, "gamma": 0.995, "n_steps": 2048, "batch_size": 512, "gae_lambda": 0.98, "ent_coef": 0.001, "clip_range": 0.1, "total_timesteps": 60000},
]

A2C_SWEEP = [
    {"run_name": "a2c_run_01", "learning_rate": 7e-4, "gamma": 0.99, "n_steps": 5, "ent_coef": 0.01, "vf_coef": 0.25, "total_timesteps": 40000},
    {"run_name": "a2c_run_02", "learning_rate": 5e-4, "gamma": 0.98, "n_steps": 10, "ent_coef": 0.015, "vf_coef": 0.3, "total_timesteps": 42000},
    {"run_name": "a2c_run_03", "learning_rate": 3e-4, "gamma": 0.99, "n_steps": 20, "ent_coef": 0.01, "vf_coef": 0.4, "total_timesteps": 45000},
    {"run_name": "a2c_run_04", "learning_rate": 2.5e-4, "gamma": 0.995, "n_steps": 20, "ent_coef": 0.02, "vf_coef": 0.25, "total_timesteps": 50000},
    {"run_name": "a2c_run_05", "learning_rate": 2e-4, "gamma": 0.99, "n_steps": 40, "ent_coef": 0.005, "vf_coef": 0.5, "total_timesteps": 52000},
    {"run_name": "a2c_run_06", "learning_rate": 1.5e-4, "gamma": 0.97, "n_steps": 5, "ent_coef": 0.03, "vf_coef": 0.2, "total_timesteps": 38000},
    {"run_name": "a2c_run_07", "learning_rate": 1.2e-4, "gamma": 0.985, "n_steps": 10, "ent_coef": 0.02, "vf_coef": 0.35, "total_timesteps": 42000},
    {"run_name": "a2c_run_08", "learning_rate": 1e-4, "gamma": 0.99, "n_steps": 30, "ent_coef": 0.01, "vf_coef": 0.45, "total_timesteps": 50000},
    {"run_name": "a2c_run_09", "learning_rate": 8e-5, "gamma": 0.995, "n_steps": 50, "ent_coef": 0.005, "vf_coef": 0.55, "total_timesteps": 55000},
    {"run_name": "a2c_run_10", "learning_rate": 5e-5, "gamma": 0.997, "n_steps": 60, "ent_coef": 0.002, "vf_coef": 0.6, "total_timesteps": 60000},
]

REINFORCE_SWEEP = [
    {"run_name": "reinforce_run_01", "learning_rate": 1e-3, "gamma": 0.95, "hidden_dim": 128, "entropy_coef": 0.01, "normalize_returns": True, "episodes": 350},
    {"run_name": "reinforce_run_02", "learning_rate": 7e-4, "gamma": 0.97, "hidden_dim": 128, "entropy_coef": 0.015, "normalize_returns": True, "episodes": 400},
    {"run_name": "reinforce_run_03", "learning_rate": 5e-4, "gamma": 0.98, "hidden_dim": 256, "entropy_coef": 0.01, "normalize_returns": True, "episodes": 450},
    {"run_name": "reinforce_run_04", "learning_rate": 3e-4, "gamma": 0.99, "hidden_dim": 256, "entropy_coef": 0.02, "normalize_returns": True, "episodes": 500},
    {"run_name": "reinforce_run_05", "learning_rate": 2e-4, "gamma": 0.995, "hidden_dim": 256, "entropy_coef": 0.005, "normalize_returns": True, "episodes": 550},
    {"run_name": "reinforce_run_06", "learning_rate": 1.5e-4, "gamma": 0.97, "hidden_dim": 128, "entropy_coef": 0.03, "normalize_returns": False, "episodes": 400},
    {"run_name": "reinforce_run_07", "learning_rate": 1e-4, "gamma": 0.985, "hidden_dim": 192, "entropy_coef": 0.02, "normalize_returns": False, "episodes": 450},
    {"run_name": "reinforce_run_08", "learning_rate": 8e-5, "gamma": 0.99, "hidden_dim": 256, "entropy_coef": 0.01, "normalize_returns": False, "episodes": 500},
    {"run_name": "reinforce_run_09", "learning_rate": 5e-5, "gamma": 0.995, "hidden_dim": 256, "entropy_coef": 0.005, "normalize_returns": False, "episodes": 550},
    {"run_name": "reinforce_run_10", "learning_rate": 3e-5, "gamma": 0.997, "hidden_dim": 256, "entropy_coef": 0.001, "normalize_returns": False, "episodes": 600},
]
