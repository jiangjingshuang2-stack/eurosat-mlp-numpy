import numpy as np


class SoftmaxCrossEntropyLoss:
    """
    Softmax + Cross Entropy Loss

    输入:
        logits: (batch_size, num_classes)
        y:      (batch_size,)  每个元素是类别编号，如 0~9

    输出:
        loss:   标量
    """

    def __init__(self):
        self.logits = None
        self.probs = None
        self.y = None
        self.batch_size = None
        self.loss = None

    def forward(self, logits: np.ndarray, y: np.ndarray) -> float:
        """
        前向传播，计算 softmax 概率和交叉熵损失
        """
        self.logits = logits
        self.y = y
        self.batch_size = logits.shape[0]

        # 数值稳定：每行减去最大值
        shifted_logits = logits - np.max(logits, axis=1, keepdims=True)

        exp_scores = np.exp(shifted_logits)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        self.probs = probs

        # 取出正确类别对应的概率
        correct_class_probs = probs[np.arange(self.batch_size), y]

        # 防止 log(0)
        eps = 1e-12
        correct_class_probs = np.clip(correct_class_probs, eps, 1.0)

        loss = -np.mean(np.log(correct_class_probs))
        self.loss = loss
        return loss

    def backward(self) -> np.ndarray:
        """
        反向传播，返回对 logits 的梯度

        dL/dlogits = probs
        对正确类别减 1
        最后除以 batch_size
        """
        dlogits = self.probs.copy()
        dlogits[np.arange(self.batch_size), self.y] -= 1.0
        dlogits /= self.batch_size
        return dlogits


if __name__ == "__main__":
    np.random.seed(42)

    # 假设 batch_size=4, num_classes=3
    logits = np.random.randn(4, 3)
    y = np.array([0, 2, 1, 1])

    criterion = SoftmaxCrossEntropyLoss()

    loss = criterion.forward(logits, y)
    print("Loss:", loss)

    dlogits = criterion.backward()
    print("dlogits shape:", dlogits.shape)
    print("dlogits:")
    print(dlogits)