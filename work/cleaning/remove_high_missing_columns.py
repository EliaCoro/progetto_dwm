def remove_high_missing_columns(data, threshold=0.5):
    """
    Rimuove le colonne con una percentuale di valori mancanti superiore alla soglia specificata.

    Returns:
    - pd.DataFrame: Il dataset con le colonne rimosse.
    - list: Lista delle colonne che sono state rimosse.
    """
    columns_with_missing_values = []
    for column in data.columns:
        if data[column].isnull().sum() > threshold * len(data):
            columns_with_missing_values.append(column)

    # Remove columns with missing values
    data_cleaned = data.drop(columns_with_missing_values, axis=1)

    return data_cleaned, columns_with_missing_values
