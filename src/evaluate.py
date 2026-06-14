import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate_model(model, loader, criterion, device):
    """
    Evaluates the model on a DataLoader.

    Returns
    -------
    dict with keys: loss, accuracy, f1, precision, recall
    """
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            input_ids, labels = batch[0].to(device), batch[1].float().to(device)

            logits = model(input_ids)
            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch[1].tolist())

    return {
        "loss": total_loss / len(loader),
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, zero_division=0),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "recall": recall_score(all_labels, all_preds, zero_division=0),
    }


def get_predictions(model, loader, device):
    """
    Returns ground-truth labels, binary predictions, and probabilities.

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    y_proba : np.ndarray  (sigmoid probabilities)
    """
    model.eval()
    y_true, y_pred, y_proba = [], [], []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch[0].to(device)
            logits = model(input_ids)
            proba = torch.sigmoid(logits).cpu().numpy()
            preds = (proba >= 0.5).astype(int)

            y_proba.extend(proba.tolist())
            y_pred.extend(preds.tolist())
            y_true.extend(batch[1].tolist())

    return np.array(y_true), np.array(y_pred), np.array(y_proba)


def plot_confusion_matrix(y_true, y_pred, labels=None):
    """Plots a confusion matrix using sklearn's ConfusionMatrixDisplay."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Matriz de Confusion", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_training_curves(history):
    """
    Plots training and validation loss and F1 curves.

    Parameters
    ----------
    history : dict
        Keys: train_loss, val_loss, train_f1, val_f1
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "o-", label="Train Loss", color="steelblue")
    axes[0].plot(epochs, history["val_loss"], "o--", label="Val Loss", color="tomato")
    axes[0].set_title("Perdida por Epoca", fontsize=13)
    axes[0].set_xlabel("Epoca")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # F1
    axes[1].plot(epochs, history["train_f1"], "o-", label="Train F1", color="steelblue")
    axes[1].plot(epochs, history["val_f1"], "o--", label="Val F1", color="tomato")
    axes[1].set_title("F1-Score por Epoca", fontsize=13)
    axes[1].set_xlabel("Epoca")
    axes[1].set_ylabel("F1-Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
