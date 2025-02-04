def remove_high_missing_fields(data, threshold=1.0):
    # Percentuale di valori nulli per riga
    row_null_percentage = data.isnull().mean(axis=1)

    # Separazione delle righe in base alla soglia
    data_cleaned = data[row_null_percentage < threshold]
    data_removed = data[row_null_percentage >= threshold]

    return data_cleaned, data_removed
