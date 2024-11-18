from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import IterativeImputer

def fill_missing_values(data, max_iter=10, random_state=0):
    columns_to_impute = data.columns[data.isnull().any()].tolist()
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