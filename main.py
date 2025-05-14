import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from src.data_loader import load_all_data
from src.features import extract_features
from src.train_model import train_gradient_boosting
from src.evaluate import evaluate_model

def main():
    Path("models").mkdir(exist_ok=True)
    
    print("Chargement des données...")
    fs1_df, ps2_df, profile_df = load_all_data()

    print("Extraction des caractéristiques...")
    features_list = []
    for idx in profile_df.index:
        pressure_values = ps2_df.iloc[idx].values
        flow_values = fs1_df.iloc[idx].values
        features = extract_features(pressure_values, flow_values)
        features_list.append(features)
    
    X = pd.DataFrame(features_list)
    
    y = (profile_df['valve_opening'] == 100).astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=2000, random_state=42
    )
    
    print("Standardisation des données...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Entraînement du modèle Gradient Boosting...")
    model = train_gradient_boosting(X_train_scaled, y_train)
    
    print("\nÉvaluation du modèle :")
    evaluate_model(model, X_test_scaled, y_test)
    
    print("\nSauvegarde du modèle...")
    joblib.dump(model, 'models/gradient_boosting_model.pkl')
    print("Modèle sauvegardé avec succès!")

if __name__ == "__main__":
    main()
