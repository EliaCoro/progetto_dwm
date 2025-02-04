import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_features_with_outliers(data):
    # Selezione delle colonne numeriche
    numeric_data = data.select_dtypes(include=['number'])
    
    # Numero delle feature
    num_features = len(numeric_data.columns)
    
    # Definizione del layout del grid plot
    n_cols = 3  # Numero delle colonne per riga
    n_rows = (num_features // n_cols) + (num_features % n_cols > 0)
    
    # Creazione dei subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    # Creazione dei boxplot per ogni colonna numerica
    for i, column in enumerate(numeric_data.columns):
        sns.boxplot(data=numeric_data, x=column, ax=axes[i], whis=1.5)
        axes[i].set_title(f"Boxplot: {column}")
        axes[i].set_xlabel("")

    # Rimozione dei subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
