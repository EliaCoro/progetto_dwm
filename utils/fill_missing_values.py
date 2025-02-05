from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

def fill_missing_values(data, columns_to_exclude=None, max_iter=10):
    if columns_to_exclude is None:
        columns_to_exclude = []
    
    # Selezione delle colonne da imputare, escludendo quelle specificate
    columns_to_impute = [col for col in data.columns if col not in columns_to_exclude and data[col].isnull().any()]
    
    # Copia del DataFrame originale per evitare modifiche in-place
    data_copy = data.copy()
    
    # Estrazione dei valori delle colonne da imputare
    X = data_copy[columns_to_impute].values
    
    # Creazione dell'imputer basato su RandomForestRegressor
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_jobs=-1),
        max_iter=max_iter
    )
    
    # Esecuzione dell'imputazione e assegnazione dei valori completati
    imputed_values = imputer.fit_transform(X)
    data_copy[columns_to_impute] = imputed_values
    
    return data_copy
