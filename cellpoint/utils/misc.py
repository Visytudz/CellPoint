import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


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


def plot_confusion_matrix(
    cm_numpy: np.ndarray,
    class_names: list[str],
    save_path: str,
    title=str,
) -> None:
    """Plot and save a confusion matrix."""
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_numpy,
        annot=True,  # show data values in each cell
        fmt="d",  # integer format
        cmap="Blues",  # color theme
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,  # draw on the axes we created
    )
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
