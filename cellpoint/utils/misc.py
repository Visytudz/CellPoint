import numpy as np


def decompose_confusion_matrix(cm) -> tuple[np.ndarray, np.ndarray]:
    """
    Reconstruct y_true and y_pred sequences from a confusion matrix.\n
    **Note: these returned sequences may not match the original order of samples.**
    """
    y_true, y_pred = [], []

    # Iterate over all cells in the confusion matrix
    for i in range(cm.shape[0]):  # true label index
        for j in range(cm.shape[1]):  # predicted label index
            count = int(cm[i, j])
            y_true += [i] * count
            y_pred += [j] * count

    return np.array(y_true), np.array(y_pred)
