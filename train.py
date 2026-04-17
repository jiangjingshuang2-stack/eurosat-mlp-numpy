import os
import json
import numpy as np
import matplotlib.pyplot as plt

from data import EuroSATDataset, train_val_test_split, get_batches, standardize_data
from model import MLP
from loss import SoftmaxCrossEntropyLoss
from optimizer import SGD


def accuracy_score(logits: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return np.mean(preds == y)


def evaluate(model, X, y, batch_size=256):
    """
    在验证集或测试集上评估 loss 和 accuracy
    """
    criterion = SoftmaxCrossEntropyLoss()

    all_losses = []
    all_logits = []

    for batch_x, batch_y in get_batches(X, y, batch_size=batch_size, shuffle=False):
        logits = model.forward(batch_x)
        loss = criterion.forward(logits, batch_y)

        all_losses.append(loss)
        all_logits.append(logits)

    all_logits = np.vstack(all_logits)
    avg_loss = float(np.mean(all_losses))
    acc = float(accuracy_score(all_logits, y))
    return avg_loss, acc


def save_training_curves(history, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # 1. train / val loss 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))
    plt.close()

    # 2. val accuracy 曲线
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "val_accuracy_curve.png"))
    plt.close()


def train_one_epoch(model, criterion, optimizer, X_train, y_train, batch_size=64):
    """
    单个 epoch 的训练
    """
    batch_losses = []
    all_preds = []
    all_labels = []

    for batch_x, batch_y in get_batches(X_train, y_train, batch_size=batch_size, shuffle=True):
        # 前向
        logits = model.forward(batch_x)

        # loss
        loss = criterion.forward(logits, batch_y)
        batch_losses.append(loss)
        all_preds.append(np.argmax(logits, axis=1))
        all_labels.append(batch_y)

        # 反向
        optimizer.zero_grad()
        dlogits = criterion.backward()
        model.backward(dlogits)

        # 更新参数
        optimizer.step()

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    train_loss = float(np.mean(batch_losses))
    train_acc = float(np.mean(all_preds == all_labels))
    return train_loss, train_acc


def main():
    # =========================
    # 1. 路径与超参数
    # =========================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, "..", "EuroSAT_RGB"))
    output_dir = os.path.join(base_dir, "outputs")
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    curve_dir = os.path.join(output_dir, "curves")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(curve_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    image_size = (32, 32)       # 先用 32x32 更快
    hidden_dim1 = 256
    hidden_dim2 = 128
    num_classes = 10
    activation = "relu"

    epochs = 100
    batch_size = 64
    lr = 0.01
    weight_decay = 1e-3
    lr_decay = 0.95
    decay_type = "multiplicative"

    np.random.seed(42)

    # =========================
    # 2. 加载数据
    # =========================
    print("Loading dataset...")
    dataset = EuroSATDataset(
        root_dir=root_dir,
        image_size=image_size,
        flatten=True,
        normalize=True
    )
    dataset.summary()

    X, y = dataset.load_data()
    print("Full dataset shape:", X.shape, y.shape)

    splits = train_val_test_split(
        X, y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True,
        seed=42
    )

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]
    X_test, y_test = splits["X_test"], splits["y_test"]

    X_train, X_val, X_test = standardize_data(X_train, X_val, X_test)
    print("Applied standardization using training-set mean and std.")

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)

    input_dim = X_train.shape[1]

    # =========================
    # 3. 初始化模型、损失、优化器
    # =========================
    model = MLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
        activation=activation
    )

    criterion = SoftmaxCrossEntropyLoss()

    optimizer = SGD(
        params=model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        decay_type=decay_type
    )

    print(model)

    # =========================
    # 4. 训练循环
    # =========================
    best_val_acc = -1.0
    best_model_path = os.path.join(checkpoint_dir, "best_model.npz")

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": []
    }

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            X_train=X_train,
            y_train=y_train,
            batch_size=batch_size
        )

        val_loss, val_acc = evaluate(
            model=model,
            X=X_val,
            y=y_val,
            batch_size=256
        )

        current_lr = optimizer.get_lr()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epoch [{epoch:02d}/{epochs}] | "
            f"lr={current_lr:.6f} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

        # 保存验证集最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save(best_model_path)
            print(f"  -> Best model saved to {best_model_path} (val_acc={best_val_acc:.4f})")

        # epoch 结束后更新学习率
        optimizer.schedule()

    # =========================
    # 5. 保存训练记录和曲线
    # =========================
    history_path = os.path.join(log_dir, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4, ensure_ascii=False)

    save_training_curves(history, curve_dir)

    print(f"\nTraining history saved to: {history_path}")
    print(f"Training curves saved to: {curve_dir}")

    # =========================
    # 6. 加载最优模型并在测试集上简单看一下
    # =========================
    print("\nLoading best model for final evaluation on test set...")
    best_model = MLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
        activation=activation
    )
    best_model.load(best_model_path)

    test_loss, test_acc = evaluate(
        model=best_model,
        X=X_test,
        y=y_test,
        batch_size=256
    )

    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
