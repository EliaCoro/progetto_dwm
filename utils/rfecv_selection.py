import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

def rfecv_selection(data: pd.DataFrame, y=None):
    # Inizializzazione di RFECV con Random Forest
    selector = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, n_jobs=-1),
    )

    # Estrazione delle feature e del target
    if y is None:
        if 'sii' not in data:
            raise ValueError("La colonna 'sii' non è presente nei dati e 'y' non è stato fornito.")
        X = data.drop(columns=['sii'])
        y = data['sii']
    else:
        X = data

    # Addestramento del selettore
    selector.fit(X, y)

    # Selezione delle feature
    selected_features = X.columns[selector.support_].tolist()
    return selector.n_features_, selected_features
