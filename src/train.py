import copy
import torch
from sklearn.metrics import f1_score


def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    num_epochs: int = 10,
    patience: int = 3,
):
    """
    Trains a binary classification model with early stopping.

    Returns
    -------
    history : dict
        Keys: train_loss, val_loss, train_f1, val_f1
    """
    history = {"train_loss": [], "val_loss": [], "train_f1": [], "val_f1": []}

    best_val_f1 = -1.0
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(1, num_epochs + 1):
        # ── Training ──────────────────────────────────────────────
        model.train()
        total_loss = 0.0
        all_preds, all_labels = [], []

        for batch in train_loader:
            input_ids, labels = batch[0].to(device), batch[1].float().to(device)

            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(batch[1].tolist())

        train_loss = total_loss / len(train_loader)
        train_f1 = f1_score(all_labels, all_preds, zero_division=0)

        # ── Validation ────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch in val_loader:
                input_ids, labels = batch[0].to(device), batch[1].float().to(device)

                logits = model(input_ids)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = (torch.sigmoid(logits) >= 0.5).long().cpu().tolist()
                val_preds.extend(preds)
                val_labels.extend(batch[1].tolist())

        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f}  Train F1: {train_f1:.4f} | "
            f"Val Loss: {val_loss:.4f}  Val F1: {val_f1:.4f}"
        )

        # ── Early stopping ────────────────────────────────────────
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping en epoca {epoch}. Mejor Val F1: {best_val_f1:.4f}")
                break

    # Restore best weights
    if best_weights is not None:
        model.load_state_dict(best_weights)

    return history
