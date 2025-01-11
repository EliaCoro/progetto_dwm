import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def PrintCorrelationMatrix(data, threshold=0.70):
    corr_matrix = data.corr()

    # Rimuovi la diagonale settando i valori a 0
    np.fill_diagonal(corr_matrix.values, 0)

    # Trova le colonne che hanno almeno una correlazione superiore alla soglia (escludendo la diagonale)
    high_corr_vars = corr_matrix.columns[(corr_matrix > threshold).any()]

    # Filtra la matrice di correlazione per mantenere solo le variabili altamente correlate
    filtered_corr_matrix = corr_matrix.loc[high_corr_vars, high_corr_vars]

    # Visualizza la matrice di correlazione filtrata
    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix (Features with |correlation| > {threshold}, excluding diagonal)")
    plt.show()