from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def grid_search_rf(X, y):
    model = RandomForestClassifier()

    param_grid = {
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 7, 9, 11],
        'min_samples_leaf': [3, 5, 7, 9, 11]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy')

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    print(f"Best parameters: {best_params}")

    return best_params