def remove_high_missing_columns(data, threshold=0.5):
    # Identificazione delle colonne con valori mancanti sopra la soglia
    columns_with_missing_values = []
    for column in data.columns:
        if data[column].isnull().sum() > threshold * len(data):
            columns_with_missing_values.append(column)

    # Rimozione delle colonne con troppi valori mancanti
    data_cleaned = data.drop(columns_with_missing_values, axis=1)

    return data_cleaned, columns_with_missing_values
