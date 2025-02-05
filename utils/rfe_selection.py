from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE

def rfe_selection(X, y, n_features):
    # Creazione del modello RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100)

    # Creazione dell'istanza di RFE per la selezione delle feature
    rfe = RFE(estimator=model, n_features_to_select=n_features)

    # Addestramento del modello
    rfe.fit(X, y)

    # Estrazione delle colonne selezionate
    selected_columns = X.columns[rfe.support_]

    return selected_columns
