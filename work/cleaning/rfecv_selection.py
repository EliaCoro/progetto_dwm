import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

def rfecv_selection(data: pd.DataFrame, y=None):
    # Initialize RFECV with cross-validation and the random forest estimator
    selector = RFECV(
        estimator=RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1),
    )
    X = data
    
    # Fit RFECV to the training data
    if y is None:
        X = data.drop(columns=['sii'])
        y = data['sii']

    selector.fit(X, y)

    # Return both the number of features and the names of the selected features
    selected_features = X.columns[selector.support_].tolist()
    return selector.n_features_, selected_features
