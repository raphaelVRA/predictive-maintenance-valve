import argparse
import logging
import os
import joblib
from sklearn.model_selection import train_test_split
from src.data_loader import load_all_data
from src.features import extract_features
from src.train_model import train_gradient_boosting
from src.evaluate import evaluate_model

def main():
    """Point d'entrée principal pour la prédiction de l'état des valves."""
    parser = argparse.ArgumentParser(description='Maintenance prédictive des valves')
    parser.add_argument('action', choices=['train', 'predict', 'evaluate'],
                       help='Action à effectuer : train, predict ou evaluate')
    args = parser.parse_args()

    # Configuration basique du logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    try:
        # Chargement des données
        logger.info("Chargement des données...")
        fs1, ps2, profile = load_all_data()
        
        # Extraction des caractéristiques
        logger.info("Extraction des caractéristiques...")
        features = extract_features(ps2, fs1, profile)
        
        # Préparation des données
        X = features.drop('valve_condition_optimal', axis=1)
        y = features['valve_condition_optimal']
        
        if args.action == 'train':
            logger.info("Début de l'entraînement du modèle...")
            # Division des données
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Entraînement avec GradientBoosting
            model = train_gradient_boosting(X_train, y_train)
            
            # Sauvegarde du modèle
            os.makedirs('models', exist_ok=True)
            model_path = os.path.join('models', 'valve_model.pkl')
            joblib.dump(model, model_path)
            logger.info(f"Modèle sauvegardé dans {model_path}")
            
            # Évaluation sur l'ensemble de test
            logger.info("Évaluation du modèle sur l'ensemble de test...")
            evaluate_model(model, X_test, y_test)
            
        elif args.action == 'evaluate':
            logger.info("Évaluation du modèle...")
            # Chargement du modèle
            model_path = os.path.join('models', 'valve_model.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError("Le modèle n'a pas été trouvé. Veuillez d'abord entraîner le modèle.")
            
            model = joblib.load(model_path)
            # Division des données pour l'évaluation
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            evaluate_model(model, X_test, y_test)
            
        elif args.action == 'predict':
            logger.info("Prédiction sur les nouvelles données...")
            # Chargement du modèle
            model_path = os.path.join('models', 'valve_model.pkl')
            if not os.path.exists(model_path):
                raise FileNotFoundError("Le modèle n'a pas été trouvé. Veuillez d'abord entraîner le modèle.")
            
            model = joblib.load(model_path)
            predictions = model.predict(X)
            
            # Affichage des résultats
            n_optimal = sum(predictions)
            total = len(predictions)
            logger.info(f"Sur {total} cycles :")
            logger.info(f"- {n_optimal} cycles prédits comme optimaux ({n_optimal/total*100:.1f}%)")
            logger.info(f"- {total-n_optimal} cycles prédits comme non-optimaux ({(total-n_optimal)/total*100:.1f}%)")

    except Exception as e:
        logger.error(f"Une erreur est survenue : {str(e)}")
        raise

if __name__ == "__main__":
    main() 