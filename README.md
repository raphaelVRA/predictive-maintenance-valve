# Maintenance Prédictive des Valves Hydrauliques

## Description du Projet
Ce projet vise à prédire la condition des valves dans un système hydraulique industriel en analysant les données de pression (PS2) et de débit (FS1).

## Structure du Projet
```
predictive-maintenance-valve/
│
├── data/                    # Données brutes
│   ├── PS2.txt             # Données de pression (100 Hz)
│   ├── FS1.txt             # Données de débit (10 Hz)
│   └── profile.txt         # Données résumées par cycle
│
├── notebooks/              # Notebooks Jupyter
│   └── 01_exploration.ipynb # Analyse exploratoire des données
│
├── src/                    # Code source
│   ├── data_loader.py      # Chargement des données
│   ├── features.py         # Extraction des caractéristiques
│   ├── train_model.py      # Entraînement des modèles
│   ├── evaluate.py         # Évaluation des modèles
│   └── utils.py           # Fonctions de visualisation et utilitaires
│
├── models/                # Modèles entraînés
├── requirements.txt       # Dépendances Python
└── main.py               # Script principal d'entraînement
```

## Fonctionnalités

### Chargement des Données (`data_loader.py`)
- Chargement des données de pression (PS2.txt)
- Chargement des données de débit (FS1.txt)
- Chargement des profils de cycle (profile.txt)

### Extraction des Caractéristiques (`features.py`)
Caractéristiques extraites pour chaque cycle :
- Pression : asymétrie, aplatissement, autocorrélation, dérivée maximale
- Débit : aplatissement de la dérivée, intégrale, asymétrie

### Entraînement (`train_model.py`)
Modèles disponibles :
- Random Forest
- Régression Logistique
- SVM
- k-NN
- Gradient Boosting (utilisé par défaut)

### Évaluation (`evaluate.py`)
- Rapport de classification détaillé
- Matrice de confusion
- Visualisation des résultats

### Visualisation (`utils.py`)
- Affichage des statistiques descriptives
- Visualisation des cycles optimaux et non-optimaux
- Distribution des caractéristiques

## Installation

1. Cloner le repository :
```bash
git clone https://github.com/raphaelVRA/predictive-maintenance-valve
```

2. Créer un environnement virtuel :
```bash
python -m venv .venv
.venv\Scripts\activate     # Windows
source .venv/bin/activate  # Unix
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

Pour entraîner le modèle :
```bash
python main.py
```

Le script va :
1. Charger les données brutes
2. Extraire les caractéristiques
3. Entraîner un modèle Gradient Boosting
4. Évaluer ses performances
5. Sauvegarder le modèle dans le dossier `models/`

## Dépendances Principales
- pandas==2.0.3
- numpy==1.24.3
- scikit-learn==1.3.0
- matplotlib==3.7.2
- seaborn==0.12.2
- scipy==1.11.3
- plotly==5.18.0

## Structure des Données

- **PS2.txt** : Données de pression échantillonnées à 100 Hz
- **FS1.txt** : Données de débit échantillonnées à 10 Hz
- **profile.txt** : Informations sur chaque cycle, incluant la condition de la valve

## Méthodologie

1. **Prétraitement** : 
   - Synchronisation des données de pression et débit
   - Extraction des caractéristiques par cycle

2. **Modélisation** :
   - Utilisation des 2000 premiers cycles pour l'entraînement
   - Évaluation sur les cycles restants

3. **Évaluation** :
   - Métriques de performance
   - Analyse des caractéristiques importantes

## Auteur
VRAIN Raphaël
