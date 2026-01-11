import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import ale_py
import numpy as np
import argparse

"""
Script d'évaluation pour les modèles PPO sur Pong.

Usage:
vu run python evaluate_PPO.py 
"""

gym.register_envs(ale_py)

# Configuration
MODEL_PATH = "./models/ppo_pong_diff3_1envs_5000000timesteps"
ENV_ID = "PongNoFrameskip-v4"
N_EPISODES = 100

print("="*70)
print("ÉVALUATION PPO SUR PONG")
print("="*70)
print(f"Modèle: {MODEL_PATH}")
print(f"Difficulté: {3}")
print(f"Nombre d'épisodes: {N_EPISODES}")
print("="*70)

# Charger le modèle
model = PPO.load(MODEL_PATH)

env_kwargs = {"difficulty": 3, "render_mode": None}
env = make_atari_env(ENV_ID, n_envs=1, env_kwargs=env_kwargs, seed=42)
env = VecFrameStack(env, n_stack=4)

# Évaluer avec evaluate_policy
print("\nÉvaluation en cours...")
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=N_EPISODES,
)

print("\n" + "="*70)
print("RÉSULTATS")
print("="*70)
print(f"Récompense moyenne sur {N_EPISODES} épisodes: {mean_reward:.2f} ± {std_reward:.2f}")
print("="*70)

