from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances d'un modèle de classification des valves.
    
    Args:
        model: Modèle entraîné à évaluer
        X_test (array-like): Features de test
        y_test (array-like): Labels de test (0: non optimal, 1: optimal)
    
    Affiche:
        - Rapport de classification (précision, rappel, f1-score)
        - Matrice de confusion visualisée avec seaborn
    """
    # Prédictions sur l'ensemble de test
    y_pred = model.predict(X_test)

    # Affichage du rapport de classification
    print("Classification Report :")
    print(classification_report(y_test, y_pred))

    # Création et affichage de la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.title('Matrice de confusion')
    plt.show()
