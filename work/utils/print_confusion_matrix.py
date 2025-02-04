from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def print_confusion_matrix(y_test_original, y_pred):
    # Matrice di confusione
    conf_matrix = confusion_matrix(y_test_original, y_pred)
    
    # Visualizzazione
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(4))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
