# Maintenance Prédictive des Valves Hydrauliques

## Description du Projet
Ce projet vise à prédire la condition des valves dans un système hydraulique industriel. L'objectif est d'aider les ingénieurs de maintenance à anticiper les défaillances en analysant les données de pression et de débit.

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
│   ├── train_model.py      # Entraînement du modèle
│   ├── evaluate.py         # Évaluation du modèle
│   └── utils.py           # Fonctions utilitaires
│
├── Dockerfile             # Configuration Docker
├── requirements.txt       # Dépendances Python
└── main.py               # Point d'entrée principal
```

## Installation

1. Cloner le repository :
```bash
git clone <url-du-repo>
cd predictive-maintenance-valve
```

2. Créer un environnement virtuel :
```bash
python -m venv .venv
source .venv/bin/activate  # Unix
# ou
.venv\Scripts\activate     # Windows
```

3. Installer les dépendances :
```bash
pip install -r requirements.txt
```

## Utilisation

### Exploration des Données
```bash
jupyter notebook notebooks/01_exploration.ipynb
```

### Entraînement du Modèle
```bash
python main.py train
```

### Prédiction
```bash
python main.py predict
```

## Docker

Construction et exécution avec Docker :
```bash
docker build -t valve-prediction .
docker run valve-prediction
```

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
[Votre Nom]

## Licence
MIT 