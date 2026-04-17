import os
import math
import numpy as np
import matplotlib.pyplot as plt

from model import MLP


def normalize_weight_image(weight_img: np.ndarray) -> np.ndarray:
    """
    将单个权重图归一化到 [0, 1]，方便可视化。
    """
    w_min = weight_img.min()
    w_max = weight_img.max()

    if abs(w_max - w_min) < 1e-12:
        return np.zeros_like(weight_img)

    weight_img = (weight_img - w_min) / (w_max - w_min)
    return weight_img


def visualize_first_layer_weights(
    checkpoint_path: str,
    image_size=(32, 32),
    hidden_dim1=256,
    hidden_dim2=128,
    num_classes=10,
    activation="relu",
    num_to_show=25,
    save_dir="./outputs/weights_vis",
    save_name="first_layer_weights.png",
):
    """
    可视化第一层权重。
    对于第一层 fc1:
        W.shape = (input_dim, hidden_dim1)

    每一列 W[:, j] 表示第 j 个隐藏神经元对输入图像的权重，
    可以 reshape 回 (H, W, 3) 进行可视化。
    """
    os.makedirs(save_dir, exist_ok=True)

    H, W = image_size
    input_dim = H * W * 3

    # 构建模型并加载参数
    model = MLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
        activation=activation,
    )
    model.load(checkpoint_path)

    # 取第一层权重
    first_layer_weights = model.fc1.W  # shape: (input_dim, hidden_dim1)

    total_neurons = first_layer_weights.shape[1]
    num_to_show = min(num_to_show, total_neurons)

    cols = 5
    rows = math.ceil(num_to_show / cols)

    plt.figure(figsize=(3 * cols, 3 * rows))

    for i in range(num_to_show):
        w = first_layer_weights[:, i]                 # (input_dim,)
        w_img = w.reshape(H, W, 3)                   # (H, W, 3)
        w_img = normalize_weight_image(w_img)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(w_img)
        plt.title(f"Neuron {i}")
        plt.axis("off")

    plt.suptitle("First Layer Weights Visualization", fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(save_dir, save_name)
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved first-layer weight visualization to: {save_path}")


def visualize_single_neuron_weight(
    checkpoint_path: str,
    neuron_idx: int,
    image_size=(32, 32),
    hidden_dim1=256,
    hidden_dim2=128,
    num_classes=10,
    activation="relu",
    save_dir="./outputs/weights_vis/single_neurons",
):
    """
    单独可视化某一个隐藏神经元对应的第一层权重图。
    方便你挑几个典型的放到报告里分析。
    """
    os.makedirs(save_dir, exist_ok=True)

    H, W = image_size
    input_dim = H * W * 3

    model = MLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
        activation=activation,
    )
    model.load(checkpoint_path)

    first_layer_weights = model.fc1.W
    total_neurons = first_layer_weights.shape[1]

    if neuron_idx < 0 or neuron_idx >= total_neurons:
        raise ValueError(f"neuron_idx 越界，应在 [0, {total_neurons - 1}] 之间")

    w = first_layer_weights[:, neuron_idx]
    w_img = w.reshape(H, W, 3)
    w_img = normalize_weight_image(w_img)

    plt.figure(figsize=(4, 4))
    plt.imshow(w_img)
    plt.title(f"Neuron {neuron_idx}")
    plt.axis("off")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"neuron_{neuron_idx:03d}.png")
    plt.savefig(save_path, dpi=200)
    plt.close()

    print(f"Saved neuron-{neuron_idx} visualization to: {save_path}")


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_dir, "outputs")
    checkpoint_path = os.path.join(output_dir, "checkpoints", "best_model.npz")

    # 这里要和 train.py 里的设置保持一致
    image_size = (32, 32)
    hidden_dim1 = 256
    hidden_dim2 = 128
    num_classes = 10
    activation = "relu"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"未找到模型权重文件: {checkpoint_path}\n"
            f"请先运行 train.py 生成 best_model.npz"
        )

    # 1. 可视化前若干个第一层神经元
    visualize_first_layer_weights(
        checkpoint_path=checkpoint_path,
        image_size=image_size,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
        activation=activation,
        num_to_show=25,
        save_dir=os.path.join(output_dir, "weights_vis"),
        save_name="first_layer_weights.png",
    )

    # 2. 单独导出几个神经元，方便报告里单独分析
    for neuron_idx in [0, 1, 2, 3, 4]:
        visualize_single_neuron_weight(
            checkpoint_path=checkpoint_path,
            neuron_idx=neuron_idx,
            image_size=image_size,
            hidden_dim1=hidden_dim1,
            hidden_dim2=hidden_dim2,
            num_classes=num_classes,
            activation=activation,
            save_dir=os.path.join(output_dir, "weights_vis", "single_neurons"),
        )


if __name__ == "__main__":
    main()
