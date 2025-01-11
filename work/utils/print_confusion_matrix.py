from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt

def print_confusion_matrix(y_test_original, y_pred):
    # Calcolare la matrice di confusione
    conf_matrix = confusion_matrix(y_test_original, y_pred)
    
    # Visualizzare la matrice di confusione
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=np.arange(4))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
