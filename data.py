import os
import random
from typing import Dict, List, Tuple, Iterator

import numpy as np
from PIL import Image


class EuroSATDataset:
    """
    读取 EuroSAT_RGB 数据集，并完成：
    1. 按类别读取图片
    2. resize 到统一大小
    3. 转为 float32 并归一化到 [0, 1]
    4. 展平成一维向量，适配 MLP
    """

    def __init__(
        self,
        root_dir: str,
        image_size: Tuple[int, int] = (32, 32),
        flatten: bool = True,
        normalize: bool = True,
        seed: int = 42,
    ):
        self.root_dir = root_dir
        self.image_size = image_size
        self.flatten = flatten
        self.normalize = normalize
        self.seed = seed

        self.class_names = self._get_class_names()
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

    def _get_class_names(self) -> List[str]:
        """
        获取所有类别文件夹名称，并按字母序排序，保证标签稳定。
        """
        class_names = [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
        ]
        class_names.sort()
        if len(class_names) == 0:
            raise ValueError(f"在路径 {self.root_dir} 下没有找到类别文件夹。")
        return class_names

    def _is_image_file(self, filename: str) -> bool:
        """
        判断文件是否是常见图片格式。
        """
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        return filename.lower().endswith(valid_exts)

    def _load_single_image(self, image_path: str) -> np.ndarray:
        """
        读取单张图片，转 RGB，resize，转 numpy。
        """
        img = Image.open(image_path).convert("RGB")
        img = img.resize(self.image_size)
        img = np.asarray(img, dtype=np.float32)

        if self.normalize:
            img = img / 255.0

        if self.flatten:
            img = img.reshape(-1)

        return img

    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        读取整个数据集，返回：
        X: [N, D] 或 [N, H, W, C]
        y: [N]
        """
        X_list = []
        y_list = []

        for class_name in self.class_names:
            class_dir = os.path.join(self.root_dir, class_name)
            label = self.class_to_idx[class_name]

            filenames = sorted(os.listdir(class_dir))
            for fname in filenames:
                if not self._is_image_file(fname):
                    continue

                image_path = os.path.join(class_dir, fname)
                try:
                    img_array = self._load_single_image(image_path)
                    X_list.append(img_array)
                    y_list.append(label)
                except Exception as e:
                    print(f"[Warning] 跳过损坏图片: {image_path}, 原因: {e}")

        if len(X_list) == 0:
            raise ValueError("没有成功读取到任何图片，请检查数据路径和图片格式。")

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int64)

        return X, y

    def summary(self) -> None:
        """
        打印数据集类别信息。
        """
        print("=== EuroSAT Dataset Summary ===")
        print(f"Root dir: {self.root_dir}")
        print(f"Num classes: {len(self.class_names)}")
        print("Classes:")
        for class_name in self.class_names:
            print(f"  {self.class_to_idx[class_name]} -> {class_name}")
        print("===============================")


def train_val_test_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    shuffle: bool = True,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    将数据划分为 train / val / test
    返回一个字典：
    {
        "X_train": ...,
        "y_train": ...,
        "X_val": ...,
        "y_val": ...,
        "X_test": ...,
        "y_test": ...
    }
    """
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio 必须等于 1.0")

    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    X = X[indices]
    y = y[indices]

    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
    }


def get_batches(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool = True,
    seed: int = 42,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    生成 mini-batch
    """
    n_samples = X.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    for start_idx in range(0, n_samples, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def compute_mean_std(X_train: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    可选：计算训练集均值和标准差。
    如果你后续想做更标准的归一化，可以用这个函数。
    对于 flatten 后的输入，这里会返回按特征维度统计的均值和标准差。
    """
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def standardize_data(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    使用训练集统计量对 train / val / test 做标准化。
    """
    mean, std = compute_mean_std(X_train)
    X_train_std = (X_train - mean) / std
    X_val_std = (X_val - mean) / std
    X_test_std = (X_test - mean) / std
    return X_train_std, X_val_std, X_test_std


if __name__ == "__main__":
    # 默认假设目录结构为 hw1/EuroSAT_RGB 和 hw1/main。
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, "..", "EuroSAT_RGB"))

    dataset = EuroSATDataset(
        root_dir=root_dir,
        image_size=(32, 32),   # 先用 32x32 跑通更快
        flatten=True,          # MLP 需要展平
        normalize=True
    )

    dataset.summary()

    X, y = dataset.load_data()
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"X dtype: {X.dtype}")
    print(f"y dtype: {y.dtype}")

    splits = train_val_test_split(
        X, y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True,
        seed=42
    )

    print("\n=== Split Result ===")
    print(f"X_train: {splits['X_train'].shape}, y_train: {splits['y_train'].shape}")
    print(f"X_val:   {splits['X_val'].shape}, y_val:   {splits['y_val'].shape}")
    print(f"X_test:  {splits['X_test'].shape}, y_test:  {splits['y_test'].shape}")

    # 测试 batch 迭代器
    print("\n=== Test get_batches ===")
    for batch_x, batch_y in get_batches(splits["X_train"], splits["y_train"], batch_size=64):
        print(f"batch_x shape: {batch_x.shape}, batch_y shape: {batch_y.shape}")
        break
