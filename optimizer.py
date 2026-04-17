import numpy as np


class SGD:
    """
    SGD 优化器，支持：
    1. 基本 SGD 更新
    2. L2 正则化（weight decay）
    3. 学习率衰减
    """

    def __init__(
        self,
        params,
        lr: float = 0.01,
        weight_decay: float = 0.0,
        lr_decay: float = 1.0,
        decay_type: str = "multiplicative",
    ):
        """
        参数:
            params: model.parameters() 返回的参数列表
            lr: 初始学习率
            weight_decay: L2 正则强度
            lr_decay: 学习率衰减系数
            decay_type:
                - "multiplicative": lr = lr * lr_decay
                - "inverse":        lr = lr / (1 + lr_decay * epoch)
        """
        self.params = params
        self.initial_lr = lr
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_decay = lr_decay
        self.decay_type = decay_type
        self.epoch = 0

    def step(self):
        """
        执行一次参数更新
        """
        for p in self.params:
            param = p["param"]
            grad = p["grad"]
            name = p.get("name", "")

            # 对权重做 L2 正则，一般不对偏置做
            if self.weight_decay > 0 and name == "W":
                grad = grad + 2.0 * self.weight_decay * param

            # SGD 更新
            param -= self.lr * grad

    def zero_grad(self):
        """
        将所有梯度清零
        注意：
        你当前这套实现里，每次 backward 都会直接覆盖 dW/db，
        所以严格来说这个函数不是必须的。
        但保留它是个好习惯，后面 train.py 调用也更规范。
        """
        for p in self.params:
            p["grad"][...] = 0.0

    def schedule(self):
        """
        每个 epoch 结束后调用一次，更新学习率
        """
        self.epoch += 1

        if self.decay_type == "multiplicative":
            self.lr = self.lr * self.lr_decay

        elif self.decay_type == "inverse":
            self.lr = self.initial_lr / (1.0 + self.lr_decay * self.epoch)

        else:
            raise ValueError(f"不支持的 decay_type: {self.decay_type}")

    def get_lr(self) -> float:
        """
        返回当前学习率
        """
        return self.lr

    def state_dict(self):
        """
        返回优化器状态，方便保存训练信息
        """
        return {
            "initial_lr": self.initial_lr,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "lr_decay": self.lr_decay,
            "decay_type": self.decay_type,
            "epoch": self.epoch,
        }

    def load_state_dict(self, state):
        """
        加载优化器状态
        """
        self.initial_lr = state["initial_lr"]
        self.lr = state["lr"]
        self.weight_decay = state["weight_decay"]
        self.lr_decay = state["lr_decay"]
        self.decay_type = state["decay_type"]
        self.epoch = state["epoch"]


if __name__ == "__main__":
    np.random.seed(42)

    # 构造假的参数，模拟 model.parameters() 的输出格式
    W = np.random.randn(4, 3).astype(np.float32)
    b = np.zeros((1, 3), dtype=np.float32)

    dW = np.random.randn(4, 3).astype(np.float32)
    db = np.random.randn(1, 3).astype(np.float32)

    params = [
        {"param": W, "grad": dW, "name": "W"},
        {"param": b, "grad": db, "name": "b"},
    ]

    optimizer = SGD(
        params=params,
        lr=0.1,
        weight_decay=1e-4,
        lr_decay=0.95,
        decay_type="multiplicative",
    )

    print("Before step:")
    print("lr =", optimizer.get_lr())
    print("W[0,0] =", W[0, 0])
    print("b[0,0] =", b[0, 0])

    optimizer.step()

    print("\nAfter step:")
    print("lr =", optimizer.get_lr())
    print("W[0,0] =", W[0, 0])
    print("b[0,0] =", b[0, 0])

    optimizer.schedule()
    print("\nAfter schedule:")
    print("lr =", optimizer.get_lr())