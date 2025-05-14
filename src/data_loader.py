import os
import pandas as pd

# Obtenir le chemin absolu vers la racine du projet
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_data_path(filename):
    """Construit le chemin absolu d'un fichier dans le dossier data/."""
    return os.path.join(PROJECT_ROOT, "data", filename)

def load_fs1(sep='\t'):
    """Charge les données de débit FS1 mesurées à 10 Hz."""
    path = get_data_path("FS1.txt")
    return pd.read_csv(path, sep=sep, engine='python')

def load_ps2(sep='\t'):
    """Charge les données de pression PS2 mesurées à 100 Hz."""
    path = get_data_path("PS2.txt")
    return pd.read_csv(path, sep=sep, engine='python')

def load_profile(sep='\t'):
    """Charge les données de profil résumant chaque cycle de production."""
    path = get_data_path("profile.txt")
    df = pd.read_csv(path, sep=sep, engine='python')
    df.columns = ['colonne_1', 'valve_opening', 'colonne_3', 'colonne_4', 'colonne_5']
    return df

def load_all_data(sep='\t'):
    """Charge toutes les données sous forme de DataFrames."""
    return load_fs1(sep), load_ps2(sep), load_profile(sep)


