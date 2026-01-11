import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import ale_py
import os
import argparse
from pathlib import Path

"""
uv run train.py --n_envs 1
uv run train.py --n_envs 8
uv run train.py --n_envs 16
uv run train.py --n_envs 32
uv run train.py --n_envs 64
uv run train.py --n_envs 128
uv run train.py --n_envs 256
uv run train.py --n_envs 512
uv run train.py --n_envs 1024
"""

gym.register_envs(ale_py)

parser = argparse.ArgumentParser(description="Train PPO on Pong with specified difficulty")
parser.add_argument("--n_envs", type=int, default=16, help="Number of parallel environments")

n_envs = parser.parse_args().n_envs
save_path = "./models"
log_path = "./logs"
difficulty = 3
total_timesteps = 5_000_000
name_env = "PongNoFrameskip-v4"
model_name = f"ppo_pong_diff{difficulty}_{n_envs}envs_{total_timesteps}timesteps"

env_kwargs = {"difficulty": difficulty, "render_mode": None}
env = make_atari_env(name_env, n_envs=n_envs, env_kwargs=env_kwargs, seed=42)
env = VecFrameStack(env, n_stack=4)

# Créer les dossiers nécessaires
Path(save_path).mkdir(parents=True, exist_ok=True)
Path(log_path).mkdir(parents=True, exist_ok=True)

model_path = os.path.join(save_path, model_name)

print("="*70)
print(f"ENTRAÎNEMENT PONG - Agent PPO")
print("="*70)
print(f"Modèle: {model_name}")
print(f"Difficulté adversaire: {difficulty}")
print(f"Environnements parallèles: {n_envs}")
print(f"Total timesteps: {total_timesteps:,}")

# Initialiser l'agent PPO 
model = PPO(
    "CnnPolicy",
    env,
    verbose=0,
    tensorboard_log=log_path,
    device="auto",

)

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


