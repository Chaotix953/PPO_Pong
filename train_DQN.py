import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import ale_py
import os
import argparse
from pathlib import Path

"""
Script d'entraînement DQN sur Pong - Structure similaire à train_PPO.py

Usage:
uv run python train_DQN.py --n_envs 8
"""

gym.register_envs(ale_py)

parser = argparse.ArgumentParser(description="Train DQN on Pong with specified difficulty")
parser.add_argument("--n_envs", type=int, default=16, help="Number of parallel environments")
parser.add_argument("--difficulty", type=int, default=3, choices=[0, 1, 2, 3], help="Difficulty level (0-3)")
parser.add_argument("--timesteps", type=int, default=8_000_000, help="Total training timesteps")
parser.add_argument("--learning_starts", type=int, default=10_000, help="Steps before learning starts")

args = parser.parse_args()

n_envs = args.n_envs
save_path = "./models"
log_path = "./logs"
difficulty = args.difficulty
total_timesteps = args.timesteps
name_env = "PongNoFrameskip-v4"
model_name = f"dqn_pong_diff{difficulty}_{n_envs}envs_{total_timesteps}timesteps"

env_kwargs = {"difficulty": difficulty, "render_mode": None}
env = make_atari_env(name_env, n_envs=n_envs, env_kwargs=env_kwargs, seed=42)
env = VecFrameStack(env, n_stack=4)

# Créer les dossiers nécessaires
Path(save_path).mkdir(parents=True, exist_ok=True)
Path(log_path).mkdir(parents=True, exist_ok=True)

model_path = os.path.join(save_path, model_name)

print("="*70)
print(f"ENTRAÎNEMENT PONG - Agent DQN")
print("="*70)
print(f"Modèle: {model_name}")
print(f"Difficulté adversaire: {difficulty}")
print(f"Environnements parallèles: {n_envs}")
print(f"Total timesteps: {total_timesteps:,}")
print("="*70)

# Initialiser l'agent DQN avec hyperparamètres optimisés pour Atari
model = DQN(
    "CnnPolicy",
    env,
    verbose=1,
    tensorboard_log=log_path,
    device="auto",
    buffer_size=400_000
)

print("\nDébut de l'entraînement...")
print("Pour visualiser les métriques: tensorboard --logdir=" + log_path)
print()

# Entraîner l'agent
model.learn(
    total_timesteps=total_timesteps,
    progress_bar=True,
    tb_log_name=model_name
)

# Fermer les environnements
env.close()

# Sauvegarder le modèle final
final_path = os.path.join(save_path, model_name)
model.save(final_path)

print(f"\nModèle final sauvegardé: {final_path}")
