def remove_high_missing_fields(data, threshold=1.0):
    """
    Rimuove le righe che hanno una percentuale di valori nulli superiore alla soglia specificata
    e restituisce sia il dataset pulito che le righe eliminate.

    Args:
    - data (pd.DataFrame): Il dataset su cui eseguire l'operazione.
    - threshold (float): La soglia di percentuale di valori nulli per riga (default 1.0, ovvero 100%).

    Returns:
    - pd.DataFrame: Il dataset con le righe rimosse.
    - pd.DataFrame: Il dataset con le righe eliminate.
    """
    # Calcola la percentuale di valori nulli per ogni riga
    row_null_percentage = data.isnull().mean(axis=1)

    # Separa le righe che hanno una percentuale di valori nulli maggiore di `threshold`
    data_cleaned = data[row_null_percentage <= threshold]
    data_removed = data[row_null_percentage > threshold]

    return data_cleaned, data_removed
