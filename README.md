# Pong - Apprentissage par Renforcement

Projet d'apprentissage par renforcement appliqué au jeu Atari Pong, utilisant les algorithmes DQN et PPO.

## Description

Ce projet implémente et compare deux algorithmes d'apprentissage par renforcement pour maîtriser le jeu Pong:
- **DQN** (Deep Q-Network)
- **PPO** (Proximal Policy Optimization)

Le projet inclut également des outils de visualisation et d'analyse des performances des agents.

## Prérequis

- Python 3.9 ou supérieur
- [uv](https://docs.astral.sh/uv/) (gestionnaire de paquets Python) ou pip

## Installation

### Option 1: Avec uv (recommandé)

```bash
# Installer uv si ce n'est pas déjà fait
curl -LsSf https://astral.sh/uv/install.sh | sh

# Cloner le projet
git clone <votre-repo>
cd pong

# Créer l'environnement virtuel et installer les dépendances
uv sync
```

### Option 2: Avec pip

```bash
# Cloner le projet
git clone <votre-repo>
cd pong

# Créer un environnement virtuel
uv venv .venv
source .venv/bin/activate  # Sur Windows: .venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
ou
uv sync
```

## Utilisation

### Entraînement

#### Entraîner un agent PPO

```bash
# Avec paramètres par défaut (16 environnements parallèles, difficulté 3)
uv run python train_PPO.py

# Avec personnalisation
uv run python train_PPO.py --n_envs 8

# Exemples avec différentes configurations
uv run python train_PPO.py --n_envs 1     # 1 environnement
uv run python train_PPO.py --n_envs 32    # 32 environnements parallèles
uv run python train_PPO.py --n_envs 128   # 128 environnements parallèles
```

#### Entraîner un agent DQN

```bash
# Avec paramètres par défaut
uv run python train_DQN.py

# Avec personnalisation
uv run python train_DQN.py --n_envs 8 --difficulty 3 --timesteps 8000000

# Options disponibles:
# --n_envs: Nombre d'environnements parallèles (défaut: 16)
# --difficulty: Niveau de difficulté de l'adversaire 0-3 (défaut: 3)
# --timesteps: Nombre total de timesteps (défaut: 8000000)
# --learning_starts: Steps avant de commencer l'apprentissage (défaut: 10000)
```

### Évaluation

#### Évaluer un modèle PPO

```bash
uv run python evaluate_PPO.py
```

#### Évaluer un modèle DQN

```bash
uv run python evaluate_DQN.py
```

Les scripts d'évaluation permettent de:
- Tester les performances d'un modèle entraîné
- Visualiser le jeu en temps réel
- Enregistrer des vidéos des parties
- Calculer les statistiques de performance

### Visualisation

#### Visualiser les activations CNN

```bash
uv run python visualise_cnn_activations.py
```

Génère des visualisations des activations des couches convolutionnelles du réseau de neurones.

#### Visualiser les trajectoires

```bash
uv run python visualise_trajectories.py
```

Crée des visualisations des trajectoires de l'agent pendant le jeu.

#### Traiter les graphiques

```bash
uv run python process_graphics.py
```

Génère des graphiques d'analyse des performances à partir des métriques collectées.

### Monitoring avec TensorBoard

```bash
# Lancer TensorBoard pour visualiser l'entraînement en temps réel
tensorboard --logdir=./logs

# Ouvrir dans le navigateur: http://localhost:6006
```

## Paramètres d'Entraînement

### PPO
- **Timesteps par défaut**: 5,000,000
- **Environnements parallèles**: 16 (configurable)
- **Difficulté**: 3 (maximale)
- **Frame stacking**: 4 frames

### DQN
- **Timesteps par défaut**: 8,000,000
- **Environnements parallèles**: 16 (configurable)
- **Difficulté**: 3 (configurable: 0-3)
- **Learning starts**: 10,000 steps
- **Frame stacking**: 4 frames

## Résultats

Les modèles entraînés sont sauvegardés dans le dossier [models/](models/) au format `.zip`.

Les logs d'entraînement sont disponibles dans [logs/](logs/) et peuvent être visualisés avec TensorBoard.

Les métriques sont enregistrées dans [metrics.csv](metrics.csv).

## Dépendances Principales

- `gymnasium[atari]` - Environnements Atari
- `stable-baselines3` - Algorithmes RL (PPO, DQN)
- `torch` - Framework deep learning
- `ale-py` - Arcade Learning Environment
- `tensorboard` - Visualisation des métriques
- `matplotlib` & `seaborn` - Visualisation graphique
- `opencv-python` - Traitement d'images
- `pandas` - Analyse de données

## Notes Techniques

### Parallélisation
Le nombre d'environnements parallèles (`n_envs`) affecte:
- **Vitesse d'entraînement**: Plus d'environnements = plus rapide
- **Utilisation mémoire**: Plus d'environnements = plus de RAM nécessaire
- **Performance**: Trouver un équilibre selon votre matériel

### GPU
Le projet utilise PyTorch et bénéficie d'un GPU CUDA si disponible. L'entraînement se fera automatiquement sur GPU si détecté.

### Difficulté de l'adversaire
Le paramètre `difficulty` (0-3) contrôle la force de l'adversaire IA du jeu Pong:
- 0: Très facile
- 1: Facile
- 2: Moyen
- 3: Difficile (par défaut)

## Licence

Ce projet utilise des bibliothèques open-source. Consultez les licences individuelles des dépendances pour plus d'informations.

## Auteur

Projet réalisé dans le cadre d'un travail sur l'apprentissage par renforcement appliqué aux jeux Atari.
