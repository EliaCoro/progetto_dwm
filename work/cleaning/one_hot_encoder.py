import numpy as np
import pandas as pd
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import OneHotEncoder

def OneHotEncoderFunction(data: pd.DataFrame) -> pd.DataFrame:
    categorical_cols = data.select_dtypes(include=['object']).columns
    numerical_cols = np.setdiff1d(np.array(data.columns), categorical_cols)

    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoder.fit(data[categorical_cols])
    data_encoded = pd.DataFrame(
        encoder.fit_transform(data[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=data.index
    )

    result = pd.concat([data[numerical_cols], data_encoded], axis=1)

    return result