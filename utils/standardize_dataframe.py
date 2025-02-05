import pandas as pd
from sklearn.preprocessing import StandardScaler

def standardize_dataframe(df):
    # Selezione delle colonne numeriche (escludendo 'sii')
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    numeric_columns = [col for col in numeric_columns if col != 'sii']

    # Copia dei valori originali di 'sii'
    sii_original = df['sii'].copy()

    # Standardizzazione delle colonne numeriche
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

    # Ripristino dei valori di 'sii'
    df['sii'] = sii_original

    return df
