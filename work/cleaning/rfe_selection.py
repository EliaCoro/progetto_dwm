from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE


def rfe_selection(X, y, n_features):
    model = RandomForestClassifier(n_estimators=100)

    # Eseguiamo RFE per ridurre il numero di caratteristiche
    rfe = RFE(estimator=model, n_features_to_select=n_features)

    # Fitting del modello
    rfe.fit(X, y)

    # Otteniamo le colonne selezionate
    selected_columns = X.columns[rfe.support_]

    return selected_columns