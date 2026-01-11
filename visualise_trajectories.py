import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import ale_py
import numpy as np
import matplotlib.pyplot as plt  # Nécessaire pour le graphique

"""
Script d'évaluation avec analyse graphique pour Pong.
"""

gym.register_envs(ale_py)

# Configuration
MODEL_PATH = "./models/ppo_pong_diff3_8envs_5000000timesteps"
ENV_ID = "PongNoFrameskip-v4"
TIMESTEPS_TO_PLOT = 400  # Nombre de frames à analyser

print("="*70)
print("ANALYSE GRAPHIQUE PPO SUR PONG")
print("="*70)

# 1. Chargement du modèle
print(f"Chargement du modèle depuis : {MODEL_PATH}")
try:
    model = PPO.load(MODEL_PATH)
except FileNotFoundError:
    print("Erreur : Le fichier modèle est introuvable. Vérifiez le chemin.")
    exit()

# 2. Création de l'environnement
# Note : render_mode="rgb_array" est suffisant pour l'analyse, "human" ralentirait trop la boucle si on veut juste les données
env_kwargs = {"difficulty": 3} 
env = make_atari_env(ENV_ID, n_envs=1, env_kwargs=env_kwargs, seed=42)
env = VecFrameStack(env, n_stack=4)

# Listes pour stocker les données
history_left_paddle = []
history_right_paddle = []
history_ball_y = []

print(f"\nExécution de la simulation sur {TIMESTEPS_TO_PLOT} timesteps...")

# 3. Boucle de simulation manuelle
obs = env.reset()

for t in range(TIMESTEPS_TO_PLOT):
    # Prédiction de l'action
    action, _ = model.predict(obs, deterministic=True)
    
    # Exécution de l'action
    obs, rewards, dones, infos = env.step(action)
    
    # --- EXTRACTION DE LA RAM ---
    # Nous devons accéder à l'environnement brut (unwrapped) à l'intérieur du VecEnv
    # env.envs[0] accède au premier environnement du vecteur
    # .unwrapped permet d'ignorer les wrappers (comme FrameStack) pour atteindre l'émulateur ALE
    unwrapped_env = env.envs[0].unwrapped
    ram = unwrapped_env.ale.getRAM()

    # Indices RAM spécifiques à Pong (Atari 2600)
    # 21: Y Raquette Ennemi (Gauche)
    # 51: Y Raquette Joueur (Droite - Notre Agent)
    # 54: Y Balle
    history_left_paddle.append(ram[21])
    history_right_paddle.append(ram[51])
    history_ball_y.append(ram[54])

    if dones[0]:
        print(f"L'épisode s'est terminé au timestep {t}. Reset automatique.")
        obs = env.reset()

env.close()

# 4. Génération du graphique
print("\nGénération du graphique 'pong_analysis.png'...")

plt.figure(figsize=(12, 6))

plt.gca().invert_yaxis()  # Inverser l'axe Y pour correspondre à la représentation visuelle
# Tracer les courbes
plt.plot(history_left_paddle, label='Raquette Gauche', color='red', alpha=0.6)
plt.plot(history_right_paddle, label='Raquette Droite (Agent PPO)', color='blue', linewidth=2)
plt.plot(history_ball_y, label='Balle', color='green', linestyle='--',alpha=0.8)

# Mise en forme
plt.title(f"Analyse des positions sur {TIMESTEPS_TO_PLOT} frames")
plt.xlabel("Timesteps (Frames)")
plt.ylabel("Position Y (Valeur RAM 0-255)")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

# Sauvegarder
output_file = "pong_analysis.png"
plt.savefig(output_file)
print(f"Graphique sauvegardé sous : {output_file}")
plt.show() # Affiche le graphique si vous êtes dans un environnement qui le supporte