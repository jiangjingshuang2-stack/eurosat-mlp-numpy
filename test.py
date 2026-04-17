import os
import numpy as np
import matplotlib.pyplot as plt

from data import EuroSATDataset, train_val_test_split, get_batches, standardize_data
from model import MLP


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(y_true == y_pred))


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm: np.ndarray, class_names):
    print("\n=== Confusion Matrix ===")
    header = "true\\pred".ljust(18)
    for name in class_names:
        header += name[:12].ljust(14)
    print(header)

    for i, row in enumerate(cm):
        line = class_names[i][:12].ljust(18)
        for val in row:
            line += str(val).ljust(14)
        print(line)


def evaluate_model(model, X_test, y_test, batch_size=256):
    all_preds = []
    all_labels = []

    for batch_x, batch_y in get_batches(X_test, y_test, batch_size=batch_size, shuffle=False):
        logits = model.forward(batch_x)
        preds = np.argmax(logits, axis=1)

        all_preds.append(preds)
        all_labels.append(batch_y)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    acc = accuracy_score(y_true, y_pred)
    return y_true, y_pred, acc


def save_misclassified_examples(
    X_test: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names,
    image_size=(32, 32),
    save_dir="./outputs/error_analysis",
    max_examples=12
):
    """
    保存部分错分样本，供报告里的 Error Analysis 使用
    注意：这里假设 X_test 是 flatten 后的图像向量，并且像素范围在 [0,1]
    """
    os.makedirs(save_dir, exist_ok=True)

    wrong_indices = np.where(y_true != y_pred)[0]
    if len(wrong_indices) == 0:
        print("No misclassified samples found.")
        return

    num_to_save = min(max_examples, len(wrong_indices))

    for i, idx in enumerate(wrong_indices[:num_to_save]):
        img = X_test[idx].reshape(image_size[0], image_size[1], 3)
        img = np.clip(img, 0.0, 1.0)

        true_label = class_names[y_true[idx]]
        pred_label = class_names[y_pred[idx]]

        plt.figure(figsize=(3, 3))
        plt.imshow(img)
        plt.title(f"True: {true_label}\nPred: {pred_label}")
        plt.axis("off")
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"misclassified_{i:02d}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"Saved {num_to_save} misclassified examples to: {save_dir}")


def main():
    # =========================
    # 1. 路径与参数
    # =========================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, "..", "EuroSAT_RGB"))
    output_dir = os.path.join(base_dir, "outputs")
    checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.npz")
    error_analysis_dir = os.path.join(output_dir, "error_analysis")

    image_size = (32, 32)
    hidden_dim1 = 256
    hidden_dim2 = 128
    num_classes = 10
    activation = "relu"

    batch_size = 256
    seed = 42

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

    splits = train_val_test_split(
        X, y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True,
        seed=seed
    )

    X_test_raw = splits["X_test"]
    _, _, X_test = standardize_data(
        splits["X_train"],
        splits["X_val"],
        X_test_raw,
    )
    y_test = splits["y_test"]

    print("Test set:", X_test.shape, y_test.shape)

    # =========================
    # 3. 构建模型并加载权重
    # =========================
    input_dim = X_test.shape[1]

    model = MLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
        activation=activation
    )

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"未找到模型权重文件: {checkpoint_path}\n"
            f"请先运行 train.py 生成 best_model.npz"
        )

    model.load(checkpoint_path)
    print(f"Loaded model from: {checkpoint_path}")

    # =========================
    # 4. 在测试集上评估
    # =========================
    y_true, y_pred, test_acc = evaluate_model(
        model=model,
        X_test=X_test,
        y_test=y_test,
        batch_size=batch_size
    )

    print(f"\nTest Accuracy: {test_acc:.4f}")

    # =========================
    # 5. 混淆矩阵
    # =========================
    cm = confusion_matrix(y_true, y_pred, num_classes=num_classes)
    print_confusion_matrix(cm, dataset.class_names)

    # 也保存一份 .npy，后面报告里可以再画图
    os.makedirs(output_dir, exist_ok=True)
    confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.npy")
    np.save(confusion_matrix_path, cm)
    print(f"\nConfusion matrix saved to: {confusion_matrix_path}")

    # =========================
    # 6. 保存错分样本
    # =========================
    save_misclassified_examples(
        X_test=X_test_raw,
        y_true=y_true,
        y_pred=y_pred,
        class_names=dataset.class_names,
        image_size=image_size,
        save_dir=error_analysis_dir,
        max_examples=12
    )


if __name__ == "__main__":
    main()
