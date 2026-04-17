import numpy as np


class Linear:
    """
    全连接层:
        out = x @ W + b

    输入:
        x: (batch_size, in_features)

    输出:
        out: (batch_size, out_features)
    """

    def __init__(self, in_features: int, out_features: int, weight_scale=None):
        self.in_features = in_features
        self.out_features = out_features

        # 参数。默认使用 He initialization，更适合 ReLU MLP。
        if weight_scale is None:
            weight_scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features) * weight_scale
        self.b = np.zeros((1, out_features), dtype=np.float32)

        # 缓存 forward 的输入，供 backward 使用
        self.x = None

        # 梯度
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        前向传播
        """
        self.x = x
        out = x @ self.W + self.b
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        反向传播

        参数:
            dout: 上游梯度, shape = (batch_size, out_features)

        返回:
            dx: 对输入 x 的梯度, shape = (batch_size, in_features)
        """
        # 对参数求梯度
        self.dW[...] = self.x.T @ dout
        self.db[...] = np.sum(dout, axis=0, keepdims=True)

        # 对输入求梯度
        dx = dout @ self.W.T
        return dx

    def parameters(self):
        """
        返回参数和对应梯度，方便优化器统一更新
        """
        return [
            {"param": self.W, "grad": self.dW, "name": "W"},
            {"param": self.b, "grad": self.db, "name": "b"},
        ]


class ReLU:
    """
    ReLU 激活函数:
        out = max(0, x)
    """

    def __init__(self):
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * (self.x > 0)
        return dx

    def parameters(self):
        return []


class Sigmoid:
    """
    Sigmoid 激活函数:
        out = 1 / (1 + exp(-x))
    """

    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = 1.0 / (1.0 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * self.out * (1.0 - self.out)
        return dx

    def parameters(self):
        return []


class Tanh:
    """
    Tanh 激活函数:
        out = tanh(x)
    """

    def __init__(self):
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = np.tanh(x)
        self.out = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * (1.0 - self.out ** 2)
        return dx

    def parameters(self):
        return []


def get_activation(name: str):
    """
    根据名字返回激活函数实例
    """
    name = name.lower()
    if name == "relu":
        return ReLU()
    elif name == "sigmoid":
        return Sigmoid()
    elif name == "tanh":
        return Tanh()
    else:
        raise ValueError(f"不支持的激活函数: {name}")


if __name__ == "__main__":
    np.random.seed(42)

    # 测试 Linear
    x = np.random.randn(4, 6)   # batch_size=4, in_features=6
    linear = Linear(6, 3)

    out = linear.forward(x)
    print("Linear forward output shape:", out.shape)   # (4, 3)

    dout = np.random.randn(4, 3)
    dx = linear.backward(dout)
    print("Linear backward dx shape:", dx.shape)       # (4, 6)
    print("Linear dW shape:", linear.dW.shape)         # (6, 3)
    print("Linear db shape:", linear.db.shape)         # (1, 3)

    # 测试 ReLU
    relu = ReLU()
    out_relu = relu.forward(out)
    dx_relu = relu.backward(dout)
    print("ReLU forward output shape:", out_relu.shape)
    print("ReLU backward dx shape:", dx_relu.shape)

    # 测试 Sigmoid
    sigmoid = Sigmoid()
    out_sigmoid = sigmoid.forward(out)
    dx_sigmoid = sigmoid.backward(dout)
    print("Sigmoid forward output shape:", out_sigmoid.shape)
    print("Sigmoid backward dx shape:", dx_sigmoid.shape)

    # 测试 Tanh
    tanh = Tanh()
    out_tanh = tanh.forward(out)
    dx_tanh = tanh.backward(dout)
    print("Tanh forward output shape:", out_tanh.shape)
    print("Tanh backward dx shape:", dx_tanh.shape)

    # 测试工厂函数
    act = get_activation("relu")
    print("Activation factory:", act.__class__.__name__)
