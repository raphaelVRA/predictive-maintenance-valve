from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def train_random_forest(X_train, y_train):
    """
    Entraîne un modèle Random Forest pour la classification des valves.
    
    Args:
        X_train (array-like): Features d'entraînement
        y_train (array-like): Labels d'entraînement (0: non optimal, 1: optimal)
        
    Returns:
        RandomForestClassifier: Modèle entraîné
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def train_logistic_regression(X_train, y_train):
    """
    Entraîne un modèle de régression logistique pour la classification des valves.
    
    Args:
        X_train (array-like): Features d'entraînement
        y_train (array-like): Labels d'entraînement (0: non optimal, 1: optimal)
        
    Returns:
        LogisticRegression: Modèle entraîné
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_svm(X_train, y_train):
    """
    Entraîne un modèle SVM pour la classification des valves.
    
    Args:
        X_train (array-like): Features d'entraînement
        y_train (array-like): Labels d'entraînement (0: non optimal, 1: optimal)
        
    Returns:
        SVC: Modèle entraîné avec probabilités activées
    """
    model = SVC(probability=True, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_knn(X_train, y_train, n_neighbors=5):
    """
    Entraîne un modèle k-NN pour la classification des valves.
    
    Args:
        X_train (array-like): Features d'entraînement
        y_train (array-like): Labels d'entraînement (0: non optimal, 1: optimal)
        n_neighbors (int, optional): Nombre de voisins à considérer. Par défaut 5
        
    Returns:
        KNeighborsClassifier: Modèle entraîné
    """
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(X_train, y_train)
    return model

def train_gradient_boosting(X_train, y_train):
    """
    Entraîne un modèle Gradient Boosting pour la classification des valves.
    
    Args:
        X_train (array-like): Features d'entraînement
        y_train (array-like): Labels d'entraînement (0: non optimal, 1: optimal)
        
    Returns:
        GradientBoostingClassifier: Modèle entraîné
    """
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model
