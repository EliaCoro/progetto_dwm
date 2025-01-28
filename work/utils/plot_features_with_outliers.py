import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_features_with_outliers(data):
    # Seleziona solo colonne numeriche
    numeric_data = data.select_dtypes(include=['number'])
    
    # Numero di feature
    num_features = len(numeric_data.columns)
    
    # Definisci il layout del grid plot
    n_cols = 3  # Numero di colonne per riga
    n_rows = (num_features // n_cols) + (num_features % n_cols > 0)
    
    # Crea i subplot
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    # Itera su ogni colonna numerica e crea il boxplot
    for i, column in enumerate(numeric_data.columns):
        sns.boxplot(data=numeric_data, x=column, ax=axes[i], whis=1.5)
        axes[i].set_title(f"Boxplot: {column}")
        axes[i].set_xlabel("")

    # Rimuovi eventuali subplot vuoti
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.tight_layout()
    plt.show()
