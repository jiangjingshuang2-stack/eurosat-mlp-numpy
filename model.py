import numpy as np
from layers import Linear, get_activation


class MLP:
    """
    三层 MLP 分类器
    结构:
        Input
        -> Linear(input_dim, hidden_dim1)
        -> Activation
        -> Linear(hidden_dim1, hidden_dim2)
        -> Activation
        -> Linear(hidden_dim2, num_classes)
        -> logits
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim1: int,
        hidden_dim2: int,
        num_classes: int,
        activation: str = "relu",
        weight_scale=None,
    ):
        self.input_dim = input_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.num_classes = num_classes
        self.activation_name = activation

        # 第一层
        self.fc1 = Linear(input_dim, hidden_dim1, weight_scale=weight_scale)
        self.act1 = get_activation(activation)

        # 第二层
        self.fc2 = Linear(hidden_dim1, hidden_dim2, weight_scale=weight_scale)
        self.act2 = get_activation(activation)

        # 第三层（输出层）
        self.fc3 = Linear(hidden_dim2, num_classes, weight_scale=weight_scale)

        # 统一管理层，方便 forward/backward
        self.layers = [
            self.fc1, self.act1,
            self.fc2, self.act2,
            self.fc3
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        输入:
            x: (batch_size, input_dim)
        输出:
            logits: (batch_size, num_classes)
        """
        out = x
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播
        输入:
            dout: loss 对输出 logits 的梯度
        输出:
            dx: loss 对输入 x 的梯度
        """
        dx = dout
        for layer in reversed(self.layers):
            dx = layer.backward(dx)
        return dx

    def parameters(self):
        """
        返回模型中所有可训练参数及其梯度
        方便优化器统一更新
        """
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def save(self, path: str):
        """
        保存模型参数到 .npz 文件
        """
        np.savez(
            path,
            fc1_W=self.fc1.W,
            fc1_b=self.fc1.b,
            fc2_W=self.fc2.W,
            fc2_b=self.fc2.b,
            fc3_W=self.fc3.W,
            fc3_b=self.fc3.b,
            input_dim=self.input_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            num_classes=self.num_classes,
            activation_name=self.activation_name,
        )

    def load(self, path: str):
        """
        从 .npz 文件加载模型参数
        """
        checkpoint = np.load(path, allow_pickle=True)

        self.fc1.W = checkpoint["fc1_W"]
        self.fc1.b = checkpoint["fc1_b"]

        self.fc2.W = checkpoint["fc2_W"]
        self.fc2.b = checkpoint["fc2_b"]

        self.fc3.W = checkpoint["fc3_W"]
        self.fc3.b = checkpoint["fc3_b"]

    def __repr__(self):
        return (
            f"MLP(input_dim={self.input_dim}, "
            f"hidden_dim1={self.hidden_dim1}, "
            f"hidden_dim2={self.hidden_dim2}, "
            f"num_classes={self.num_classes}, "
            f"activation='{self.activation_name}')"
        )


if __name__ == "__main__":
    np.random.seed(42)

    # 假设输入是一批展平后的图像向量
    batch_size = 8
    input_dim = 32 * 32 * 3   # 如果图像是 32x32 RGB
    hidden_dim1 = 256
    hidden_dim2 = 128
    num_classes = 10

    model = MLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
        activation="relu",
    )

    print(model)

    x = np.random.randn(batch_size, input_dim).astype(np.float32)

    # 前向传播
    logits = model.forward(x)
    print("logits shape:", logits.shape)   # (8, 10)

    # 假设上游梯度
    dout = np.random.randn(batch_size, num_classes).astype(np.float32)

    # 反向传播
    dx = model.backward(dout)
    print("dx shape:", dx.shape)           # (8, input_dim)

    # 查看参数数量
    params = model.parameters()
    print("number of parameter tensors:", len(params))
    for i, p in enumerate(params):
        print(
            f"param {i}: {p['name']}, "
            f"param shape={p['param'].shape}, "
            f"grad shape={p['grad'].shape}"
        )

    # 测试保存和加载
    save_path = "test_mlp_model.npz"
    model.save(save_path)
    print(f"Model saved to {save_path}")

    new_model = MLP(
        input_dim=input_dim,
        hidden_dim1=hidden_dim1,
        hidden_dim2=hidden_dim2,
        num_classes=num_classes,
        activation="relu",
    )
    new_model.load(save_path)
    print("Model loaded successfully.")
