import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3.common.env_util import make_atari_env
import ale_py
import numpy as np
import matplotlib.pyplot as plt
import torch

"""
Script d'analyse Deep Visu PPO - Pong (Corrigé & Optimisé)
"""

gym.register_envs(ale_py)

# Configuration
MODEL_PATH = "./models/ppo_pong_diff3_8envs_5000000timesteps"
ENV_ID = "PongNoFrameskip-v4"
DIFFICULTY = 3

# Noms des actions Pong (standard ALE)
ACTION_NAMES = ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'R_FIRE', 'L_FIRE']

print("="*70)
print("ANALYSE DE L'ATTENTION DU CNN (SALIENCY MAP)")
print("="*70)

# 1. Chargement
try:
    model = PPO.load(MODEL_PATH)
except FileNotFoundError:
    print(f"Modèle introuvable à {MODEL_PATH}")
    exit()

# Création de l'environnement (identique à l'entraînement)
env_kwargs = {"difficulty": DIFFICULTY, "render_mode": "rgb_array"}
env = make_atari_env(ENV_ID, n_envs=1, env_kwargs=env_kwargs, seed=42)
env = VecFrameStack(env, n_stack=4)

# 2. Fonction pour extraire les données internes (CORRIGÉE)
def get_net_internals(model, obs):
    """
    Récupère la carte de saillance (gradient) et les probabilités d'action.
    """
    # Récupérer le tenseur sur le bon device (CPU/GPU)
    obs_tensor = model.policy.obs_to_tensor(obs)[0]
    
    # --- CORRECTION CRITIQUE ---
    # 1. Convertir en Float pour permettre le gradient
    # 2. Diviser par 255.0 car SB3 ne le fait plus automatiquement quand c'est déjà du float !
    obs_tensor = obs_tensor.float() / 255.0
    
    # Activer le gradient sur l'image
    obs_tensor.requires_grad_(True)
    
    # --- Forward Pass ---
    # On passe directement au features_extractor (l'étape de preprocessing est faite manuellement ci-dessus)
    features = model.policy.features_extractor(obs_tensor)
    latent_pi = model.policy.mlp_extractor.forward_actor(features)
    distribution = model.policy._get_action_dist_from_latent(latent_pi)
    
    probs = distribution.distribution.probs.detach().cpu().numpy()[0]
    entropy = distribution.entropy().detach().item()
    
    # --- Backward Pass ---
    action_logits = distribution.distribution.logits
    top_action = torch.argmax(action_logits)
    score = action_logits[0, top_action]
    
    model.policy.zero_grad()
    score.backward()
    
    gradient = obs_tensor.grad.abs()
    saliency_map = torch.max(gradient, dim=1)[0].squeeze().cpu().numpy()
    
    return saliency_map, probs, entropy

# 3. Boucle de capture INTELLIGENTE
print("Lancement de la simulation... Recherche de mouvements actifs.")

captured_data = [] 
obs = env.reset()
steps = 0
max_steps = 1000 # On laisse le temps au jeu de démarrer

while steps < max_steps and len(captured_data) < 4:
    # On prédit l'action
    action, _ = model.predict(obs, deterministic=True)
    action_val = action[0]
    
    # Logique de capture :
    # 1. On attend que le jeu ait "chauffé" (> 50 steps)
    # 2. On capture si l'agent décide de BOUGER (Action 2=Right ou 3=Left)
    # 3. On espace les captures d'au moins 20 frames pour la variété
    
    last_capture_step = captured_data[-1]['ts'] if captured_data else -100
    
    if steps > 54 :
        # Si l'action est un mouvement (2 ou 3) OU qu'on a attendu trop longtemps (>400 steps)
        print(f"Capture au step {steps} (Action: {ACTION_NAMES[action_val]})")
        
        # On récupère l'image réelle pour l'affichage
        rgb_frame = env.envs[0].render()
        
        # Calcul de la Saliency Map
        saliency, probs, entropy = get_net_internals(model, obs)
        
        captured_data.append({
            'ts': steps,
            'rgb': rgb_frame,
            'saliency': saliency,
            'probs': probs,
            'entropy': entropy,
            'action_taken': action_val
        })

    # Étape suivante
    obs, _, done, _ = env.step(action)
    steps += 1
    
    if done:
        obs = env.reset()

env.close()

if len(captured_data) == 0:
    print("ATTENTION : Aucune donnée capturée. L'agent n'a peut-être jamais bougé.")
    exit()

# 4. Affichage Graphique
print(f"\nGénération du graphique avec {len(captured_data)} captures...")
fig, axes = plt.subplots(len(captured_data), 3, figsize=(15, 3.5 * len(captured_data)))
plt.subplots_adjust(hspace=0.4, wspace=0.3)

# Gérer le cas où il n'y a qu'une seule ligne (numpy array 1D)
if len(captured_data) == 1:
    axes = np.expand_dims(axes, axis=0)

for i, data in enumerate(captured_data):
    # COLONNE 1 : Image du jeu
    axes[i, 0].imshow(data['rgb'])
    axes[i, 0].set_title(f"Step {data['ts']} - Action: {ACTION_NAMES[data['action_taken']]}", fontsize=10, fontweight='bold')
    axes[i, 0].axis('off')
    
    # COLONNE 2 : Saliency Map
    axes[i, 1].imshow(data['saliency'], cmap='inferno', interpolation='bilinear')
    axes[i, 1].set_title("Focus CNN", fontsize=10, fontweight='bold')
    axes[i, 1].axis('off')
    
    # COLONNE 3 : Distribution
    x_pos = np.arange(len(ACTION_NAMES))
    bars = axes[i, 2].bar(x_pos, data['probs'], color='skyblue', edgecolor='navy')
    axes[i, 2].set_ylim(0, 1.1)
    axes[i, 2].set_xticks(x_pos)
    axes[i, 2].set_xticklabels(ACTION_NAMES, rotation=45, fontsize=8)
    axes[i, 2].set_title(f"Incertitude (Entropie: {data['entropy']:.2f})", fontsize=10)
    
    # Surligner l'action choisie
    bars[data['action_taken']].set_color('orange')

output_file = "ppo_pong_deep_visu_corrected.png"
plt.savefig(output_file, dpi=150, bbox_inches='tight')
print(f"Analyse terminée. Image sauvegardée sous : {output_file}")
plt.show()