import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def one_hot_encoder(data: pd.DataFrame, columns_to_exclude=None) -> pd.DataFrame:
    if columns_to_exclude is None:
        columns_to_exclude = []

    # Selezione delle colonne categoriche e numeriche
    categorical_cols = data.select_dtypes(include=['object']).columns.difference(columns_to_exclude)
    numerical_cols = data.columns.difference(categorical_cols).difference(columns_to_exclude)

    # Creazione e addestramento dell'encoder
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoder.fit(data[categorical_cols])

    # Trasformazione delle colonne categoriche
    data_encoded = pd.DataFrame(
        encoder.transform(data[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=data.index
    )

    print("Colonne modificate:", list(categorical_cols))

    # Combina le colonne numeriche, codificate e quelle escluse
    result = pd.concat([data[numerical_cols], data[columns_to_exclude], data_encoded], axis=1)

    return result