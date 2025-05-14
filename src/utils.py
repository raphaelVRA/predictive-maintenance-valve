import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def afficher_statistiques(nom, df):
    """Affiche les statistiques descriptives pour chaque DataFrame."""
    print(f"\n--- Statistiques pour {nom} ---")
    print(f"Dimensions (lignes, colonnes) : {df.shape}")
    print("Types de variables :")
    print(df.dtypes)
    print("\nAperçu des premières lignes :")
    print(df.head())

# Sélection des cycles
def select_cycles(df_profile, n_cycles=3):
    optimal_indices = df_profile[df_profile['valve_opening'] == 100].index[:n_cycles]
    non_optimal_indices = df_profile[df_profile['valve_opening'] != 100].index[:n_cycles]
    return optimal_indices.tolist() + non_optimal_indices.tolist()

# Extraction des segments
def extract_segments(df_profile, df_pressure, df_flow, selected_indices):
    pressure_segments = []
    flow_segments = []
    labels = []

    for idx in selected_indices:
        # Accéder directement aux lignes sans nom de colonnes
        pressure_values = df_pressure.iloc[idx].values  # Accède à la ligne par index, convertie en tableau numpy
        flow_values = df_flow.iloc[idx].values  # Accède à la ligne par index, convertie en tableau numpy
        label = 'Optimale' if df_profile.loc[idx, 'valve_opening'] == 100 else 'Non optimale'

        pressure_segments.append(pressure_values)
        flow_segments.append(flow_values)
        labels.append(label)

    return pressure_segments, flow_segments, labels

# Visualisation : un graphique par cycle
def plot_data(pressure_segments, flow_segments, labels):
    # Créer une figure avec 2 lignes et 2 colonnes
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Parcourir tous les segments
    for p_segment, f_segment, label in zip(pressure_segments, flow_segments, labels):
        if label == 'Optimale':
            # Tracer les pressions optimales
            ax1.plot(p_segment, alpha=0.7, color='blue')
            # Tracer les débits optimaux
            ax3.plot(f_segment, alpha=0.7, color='blue')
        else:
            # Tracer les pressions non optimales
            ax2.plot(p_segment, alpha=0.7, color='red')
            # Tracer les débits non optimaux
            ax4.plot(f_segment, alpha=0.7, color='red')
    
    # Configuration du graphique des pressions optimales
    ax1.set_title("Pressions - Cycles Optimaux")
    ax1.set_xlabel("Temps")
    ax1.set_ylabel("Pression (PS2)")
    ax1.grid(True)
    
    # Configuration du graphique des pressions non optimales
    ax2.set_title("Pressions - Cycles Non Optimaux")
    ax2.set_xlabel("Temps")
    ax2.set_ylabel("Pression (PS2)")
    ax2.grid(True)
    
    # Configuration du graphique des débits optimaux
    ax3.set_title("Débits - Cycles Optimaux")
    ax3.set_xlabel("Temps")
    ax3.set_ylabel("Débit (FS1)")
    ax3.grid(True)
    
    # Configuration du graphique des débits non optimaux
    ax4.set_title("Débits - Cycles Non Optimaux")
    ax4.set_xlabel("Temps")
    ax4.set_ylabel("Débit (FS1)")
    ax4.grid(True)
    
    plt.tight_layout()
    plt.show()

def create_target_variable(df_profile):
    """
    Crée une variable cible binaire 'valve_condition_optimal'
    :param df_profile: DataFrame contenant les informations sur les cycles (profile)
    :return: DataFrame avec la variable cible ajoutée
    """
    # Vérifiez si la colonne 'valve_opening' existe dans le DataFrame
    if 'valve_opening' not in df_profile.columns:
        raise ValueError("La colonne 'valve_opening' n'existe pas dans le DataFrame.")

    # La variable cible est 1 si valve_opening == 100, sinon 0
    df_profile['valve_condition_optimal'] = np.where(df_profile['valve_opening'] == 100, 1, 0)
    return df_profile


def plot_feature_distribution(df_features, target_variable, feature_columns):
    """
    Affiche la distribution des caractéristiques par rapport à la variable cible.
    Utilise des KDE plots pour afficher les distributions avec une légende de couleurs.

    :param df_features: DataFrame contenant les caractéristiques extraites
    :param target_variable: Nom de la variable cible
    :param feature_columns: Liste des colonnes de caractéristiques à visualiser
    """
    # Définir les couleurs pour les classes 0 et 1
    color_map = {0: 'tab:blue', 1: 'tab:orange'}

    for feature in feature_columns:
        plt.figure(figsize=(8, 6))

        # KDE plot pour chaque classe de la variable cible avec couleur spécifique
        sns.kdeplot(df_features[df_features[target_variable] == 0][feature], fill=True,
                    label='Classe 0 (Non optimale)', color=color_map[0], alpha=0.6)
        sns.kdeplot(df_features[df_features[target_variable] == 1][feature], fill=True, label='Classe 1 (Optimale)',
                    color=color_map[1], alpha=0.6)

        # Ajouter la légende et le titre
        plt.legend(title=target_variable)

        # Titre et autres paramètres du graphique
        plt.title(f'Distribution de {feature} selon {target_variable}')
        plt.tight_layout()
        plt.show()

