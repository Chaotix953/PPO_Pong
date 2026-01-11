import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
import ale_py
import numpy as np
import argparse

"""
Script d'évaluation pour les modèles DQN sur Pong.
Similaire à evaluate.py pour PPO.

Usage:
uv run python evaluate_DQN.py --model-path models/dqn_pong_diff3_16envs_5000000timesteps
uv run python evaluate_DQN.py --model-path models/dqn_pong_diff3_8envs_5000000timesteps --episodes 50
"""

gym.register_envs(ale_py)

parser = argparse.ArgumentParser(description="Évaluer un modèle DQN entraîné sur Pong")
parser.add_argument(
    "--model-path",
    type=str,
    required=True,
    help="Chemin vers le modèle entraîné (sans l'extension .zip)"
)
parser.add_argument(
    "--episodes",
    type=int,
    default=100,
    help="Nombre d'épisodes d'évaluation"
)
parser.add_argument(
    "--difficulty",
    type=int,
    default=3,
    choices=[0, 1, 2, 3],
    help="Niveau de difficulté (0-3)"
)
parser.add_argument(
    "--render",
    action="store_true",
    help="Afficher le rendu visuel"
)

args = parser.parse_args()

# Configuration
MODEL_PATH = args.model_path
ENV_ID = "PongNoFrameskip-v4"
N_EPISODES = args.episodes
DIFFICULTY = args.difficulty

print("="*70)
print("ÉVALUATION DQN SUR PONG")
print("="*70)
print(f"Modèle: {MODEL_PATH}")
print(f"Difficulté: {DIFFICULTY}")
print(f"Nombre d'épisodes: {N_EPISODES}")
print("="*70)

# Charger le modèle
model = DQN.load(MODEL_PATH)

# Créer l'environnement d'évaluation
env_kwargs = {"render_mode": "human" if args.render else None, "difficulty": DIFFICULTY}
env = make_atari_env(ENV_ID, n_envs=1, env_kwargs=env_kwargs, seed=42)
env = VecFrameStack(env, n_stack=4)

# Évaluer avec evaluate_policy
print("\nÉvaluation en cours...")
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=N_EPISODES,
    deterministic=True
)

print("\n" + "="*70)
print("RÉSULTATS")
print("="*70)
print(f"Récompense moyenne sur {N_EPISODES} épisodes: {mean_reward:.2f} ± {std_reward:.2f}")
print("="*70)

# Évaluation détaillée avec comptage des victoires/défaites
print("\nÉvaluation détaillée avec comptage des victoires...")

episode_rewards = []
wins = 0
losses = 0
draws = 0

for episode in range(N_EPISODES):
    obs = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        episode_reward += reward[0]

    episode_rewards.append(episode_reward)

    if episode_reward > 0:
        wins += 1
    elif episode_reward < 0:
        losses += 1
    else:
        draws += 1

    if (episode + 1) % 10 == 0:
        print(f"Épisode {episode + 1}/{N_EPISODES} terminé...")

env.close()

# Afficher les statistiques détaillées
print("\n" + "="*70)
print("STATISTIQUES DÉTAILLÉES")
print("="*70)
print(f"Récompense moyenne: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
print(f"Récompense min: {np.min(episode_rewards):.2f}")
print(f"Récompense max: {np.max(episode_rewards):.2f}")
print(f"\nVictoires: {wins}/{N_EPISODES} ({wins/N_EPISODES*100:.1f}%)")
print(f"Défaites: {losses}/{N_EPISODES} ({losses/N_EPISODES*100:.1f}%)")
print(f"Matchs nuls: {draws}/{N_EPISODES} ({draws/N_EPISODES*100:.1f}%)")

if wins > losses:
    print("\n✓ L'agent DQN domine l'IA intégrée")
elif wins < losses:
    print("\n✗ L'IA intégrée est plus forte que l'agent DQN")
else:
    print("\n= Match équilibré")

print("="*70)
