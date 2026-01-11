from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def extract_tfevents(log_dir, output_file="metrics.csv"):
    """Extrait les métriques des fichiers TensorBoard et les sauvegarde en CSV."""
    # Liste tous les fichiers .tfevents dans le dossier
    event_files = []
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if "events.out.tfevents" in file:
                event_files.append(os.path.join(root, file))

    if not event_files:
        print("Aucun fichier .tfevents trouvé.")
        return None

    # Initialise un DataFrame pour stocker toutes les métriques
    all_metrics = []

    for event_file in event_files:
        # Charge les événements
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()

        # Récupère les tags (noms des métriques)
        tags = event_acc.Tags()["scalars"]

        for tag in tags:
            # Récupère les données pour chaque tag
            scalar_events = event_acc.Scalars(tag)

            for event in scalar_events:
                all_metrics.append({
                    "run": os.path.basename(os.path.dirname(event_file)),
                    "tag": tag,
                    "step": event.step,
                    "value": event.value,
                    "wall_time": event.wall_time
                })

    # Convertit en DataFrame
    df = pd.DataFrame(all_metrics)

    # Sauvegarde en CSV
    df.to_csv(output_file, index=False)
    print(f"Métriques sauvegardées dans {output_file}")

    return df


def generate_graphics(csv_file="metrics.csv", selected_runs=None, selected_metrics=None, max_steps=5_000_000, agent_names=None):
    """Génère les graphiques à partir du fichier CSV de métriques."""
    # Configuration des styles pour des graphiques plus beaux
    sns.set(style="whitegrid")
    plt.rcParams["figure.figsize"] = (10, 6)
    plt.rcParams["lines.linewidth"] = 2
    plt.rcParams["font.size"] = 12

    # Charger les données depuis le fichier CSV
    metrics_df = pd.read_csv(csv_file)

    # Configuration par défaut
    if selected_runs is None:
        selected_runs = [
            "ppo_pong_diff3_1envs_5000000timesteps_1",
            "ppo_pong_diff3_8envs_5000000timesteps_2",
            "ppo_pong_diff3_16envs_5000000timesteps_1",
            "ppo_pong_diff3_32envs_5000000timesteps_1",
            "ppo_pong_diff3_64envs_5000000timesteps_1",
            "ppo_pong_diff3_128envs_5000000timesteps_1",
        ]

    if selected_metrics is None:
        selected_metrics = [
            "rollout/ep_len_mean",
            "rollout/ep_rew_mean",
            "train/loss",
        ]

    if agent_names is None:
        agent_names = {
            "ppo_pong_diff3_1envs_5000000timesteps_1": "PPO - 1 env",
            "ppo_pong_diff3_8envs_5000000timesteps_2": "PPO - 8 envs",
            "ppo_pong_diff3_16envs_5000000timesteps_1": "PPO - 16 envs",
            "ppo_pong_diff3_32envs_5000000timesteps_1": "PPO - 32 envs",
            "ppo_pong_diff3_64envs_5000000timesteps_1": "PPO - 64 envs",
            "ppo_pong_diff3_128envs_5000000timesteps_1": "PPO - 128 envs",
        }

    # Filtrer les données
    filtered_df = metrics_df[
        (metrics_df["run"].isin(selected_runs)) &
        (metrics_df["step"] <= max_steps) &
        (metrics_df["tag"].isin(selected_metrics))
    ]

    # Appliquer le renommage après le filtrage
    filtered_df["run"] = filtered_df["run"].map(agent_names)

    # Créer un graphique pour chaque métrique
    for metric in selected_metrics:
        # Filtrer les données pour la métrique actuelle
        metric_data = filtered_df[filtered_df["tag"] == metric]

        # Créer le graphique
        plt.figure()
        sns.lineplot(
            data=metric_data,
            x="step",
            y="value",
            hue="run",
        )

        # Personnaliser le graphique
        plt.title(f"{metric}")
        plt.xlabel("Timesteps")
        plt.ylabel("Valeur")
        plt.legend(title="Agent")
        plt.grid(True, linestyle="--", alpha=0.6)

        plt.tight_layout()
        # Sauvegarder le graphique
        filename = metric.replace("/", "_")  # Remplacer les / par des _
        plt.savefig(f"{filename}.png", dpi=300, bbox_inches="tight")
        print(f"Graphique sauvegardé: {filename}.png")
        plt.close()  # Fermer la figure pour libérer la mémoire


if __name__ == "__main__":
    # Exemple d'utilisation
    log_dir = "./logs/"  # Chemin vers votre dossier de logs TensorBoard

    # Étape 1: Extraire les métriques
    print("Extraction des métriques TensorBoard...")
    metrics_df = extract_tfevents(log_dir)

    if metrics_df is not None:
        print(metrics_df.head())

        # Étape 2: Générer les graphiques
        print("\nGénération des graphiques...")
        generate_graphics()
        print("Terminé!")
