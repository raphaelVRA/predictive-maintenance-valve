import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

def afficher_statistiques(nom, df):
    """
    Affiche les statistiques descriptives d'un DataFrame.
    
    Args:
        nom (str): Nom du DataFrame à afficher dans le résumé
        df (pandas.DataFrame): DataFrame à analyser
        
    Affiche:
        - Dimensions du DataFrame
        - Types des variables
        - Aperçu des premières lignes
    """
    print(f"\n--- Statistiques pour {nom} ---")
    print(f"Dimensions (lignes, colonnes) : {df.shape}")
    print("Types de variables :")
    print(df.dtypes)
    print("\nAperçu des premières lignes :")
    print(df.head())

# Sélection des cycles
def select_cycles(df_profile, n_cycles=3):
    """
    Sélectionne un nombre égal de cycles optimaux et non-optimaux.
    
    Args:
        df_profile (pandas.DataFrame): DataFrame contenant les informations des cycles
        n_cycles (int, optional): Nombre de cycles à sélectionner pour chaque catégorie. Par défaut 3
        
    Returns:
        list: Liste des indices des cycles sélectionnés (optimaux puis non-optimaux)
    """
    optimal_indices = df_profile[df_profile['valve_opening'] == 100].index[:n_cycles]
    non_optimal_indices = df_profile[df_profile['valve_opening'] != 100].index[:n_cycles]
    return optimal_indices.tolist() + non_optimal_indices.tolist()

# Extraction des segments
def extract_segments(df_profile, df_pressure, df_flow, selected_indices):
    """
    Extrait les segments de pression et débit pour les cycles sélectionnés.
    
    Args:
        df_profile (pandas.DataFrame): DataFrame des informations par cycle
        df_pressure (pandas.DataFrame): DataFrame des données de pression
        df_flow (pandas.DataFrame): DataFrame des données de débit
        selected_indices (list): Liste des indices des cycles à extraire
        
    Returns:
        tuple: (pressure_segments, flow_segments, labels)
            - pressure_segments: Liste des segments de pression
            - flow_segments: Liste des segments de débit
            - labels: Liste des labels ('Optimale' ou 'Non optimale')
    """
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
    """
    Visualise les données de pression et débit en séparant cycles optimaux et non-optimaux.
    
    Args:
        pressure_segments (list): Liste des segments de pression
        flow_segments (list): Liste des segments de débit
        labels (list): Liste des labels ('Optimale' ou 'Non optimale')
        
    Affiche:
        Figure avec 4 sous-graphiques :
        - Pressions des cycles optimaux
        - Pressions des cycles non-optimaux
        - Débits des cycles optimaux
        - Débits des cycles non-optimaux
    """
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
    Crée une variable cible binaire pour la classification des valves.
    
    Args:
        df_profile (pandas.DataFrame): DataFrame contenant la colonne 'valve_opening'
        
    Returns:
        pandas.DataFrame: DataFrame avec la nouvelle colonne 'valve_condition_optimal'
            (1 si valve_opening == 100, 0 sinon)
            
    Raises:
        ValueError: Si la colonne 'valve_opening' n'existe pas
    """
    # Vérifiez si la colonne 'valve_opening' existe dans le DataFrame
    if 'valve_opening' not in df_profile.columns:
        raise ValueError("La colonne 'valve_opening' n'existe pas dans le DataFrame.")

    # La variable cible est 1 si valve_opening == 100, sinon 0
    df_profile['valve_condition_optimal'] = np.where(df_profile['valve_opening'] == 100, 1, 0)
    return df_profile


def plot_feature_distribution(df_features, target_variable, feature_columns):
    """
    Visualise la distribution des caractéristiques selon la variable cible.
    
    Args:
        df_features (pandas.DataFrame): DataFrame contenant les caractéristiques
        target_variable (str): Nom de la colonne cible
        feature_columns (list): Liste des colonnes de caractéristiques à visualiser
        
    Affiche:
        Pour chaque caractéristique :
        - Distribution (KDE plot) pour chaque classe
        - Légende indiquant les classes
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

