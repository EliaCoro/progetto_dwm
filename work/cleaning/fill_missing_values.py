from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

def fill_missing_values(data, columns_to_exclude=None, max_iter=10, random_state=0):
    if columns_to_exclude is None:
        columns_to_exclude = []
    columns_to_impute = [col for col in data.columns if col not in columns_to_exclude and data[col].isnull().any()]
    data_copy = data.copy()
    X = data_copy[columns_to_impute].values
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_jobs=-1),
        max_iter=max_iter,
        random_state=random_state
    )
    imputed_values = imputer.fit_transform(X)
    data_copy[columns_to_impute] = imputed_values
    
    return data_copy
