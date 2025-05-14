import os
import pandas as pd

# Obtenir le chemin absolu vers la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(filename):
    """
    Construit le chemin absolu vers un fichier dans le dossier data/.
    
    Args:
        filename (str): Nom du fichier à localiser
        
    Returns:
        str: Chemin absolu vers le fichier
    """
    return os.path.join(PROJECT_ROOT, "data", filename)

def load_fs1(sep='\t'):
    """
    Charge les données de débit FS1 mesurées à 10 Hz.
    
    Args:
        sep (str, optional): Séparateur utilisé dans le fichier. Par défaut '\t'
        
    Returns:
        pandas.DataFrame: Données de débit avec une mesure par ligne
    """
    path = get_data_path("FS1.txt")
    return pd.read_csv(path, sep=sep, engine='python')

def load_ps2(sep='\t'):
    """
    Charge les données de pression PS2 mesurées à 100 Hz.
    
    Args:
        sep (str, optional): Séparateur utilisé dans le fichier. Par défaut '\t'
        
    Returns:
        pandas.DataFrame: Données de pression avec une mesure par ligne
    """
    path = get_data_path("PS2.txt")
    return pd.read_csv(path, sep=sep, engine='python')

def load_profile(sep='\t'):
    """
    Charge les données de profil résumant chaque cycle de production.
    
    Args:
        sep (str, optional): Séparateur utilisé dans le fichier. Par défaut '\t'
        
    Returns:
        pandas.DataFrame: Données de profil avec les colonnes:
            - colonne_1: Identifiant du cycle
            - valve_opening: Pourcentage d'ouverture de la valve (100% = optimal)
            - colonne_3, colonne_4, colonne_5: Autres mesures du cycle
    """
    path = get_data_path("profile.txt")
    df = pd.read_csv(path, sep=sep, engine='python')
    df.columns = ['colonne_1', 'valve_opening', 'colonne_3', 'colonne_4', 'colonne_5']
    return df

def load_all_data(sep='\t'):
    """
    Charge l'ensemble des données du système hydraulique.
    
    Args:
        sep (str, optional): Séparateur utilisé dans les fichiers. Par défaut '\t'
        
    Returns:
        tuple: (fs1_df, ps2_df, profile_df)
            - fs1_df: DataFrame des données de débit (10 Hz)
            - ps2_df: DataFrame des données de pression (100 Hz)
            - profile_df: DataFrame des informations par cycle
    """
    return load_fs1(sep), load_ps2(sep), load_profile(sep)


