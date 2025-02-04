import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def PrintCorrelationMatrix(data, threshold=0.70):
    corr_matrix = data.corr()

    # Rimozione della diagonale
    np.fill_diagonal(corr_matrix.values, 0)

    # Selezione delle colonne con correlazione superiore alla soglia
    high_corr_vars = corr_matrix.columns[(corr_matrix > threshold).any()]

    # Filtraggio della matrice
    filtered_corr_matrix = corr_matrix.loc[high_corr_vars, high_corr_vars]

    # Visualizzazione
    plt.figure(figsize=(12, 10))
    sns.heatmap(filtered_corr_matrix, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix (Features with |correlation| > {threshold}, excluding diagonal)")
    plt.show()
