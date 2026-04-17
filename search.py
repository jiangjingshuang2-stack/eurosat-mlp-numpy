import os
import json
import itertools
import numpy as np

from data import EuroSATDataset, train_val_test_split, get_batches, standardize_data
from model import MLP
from loss import SoftmaxCrossEntropyLoss
from optimizer import SGD


def accuracy_score(logits: np.ndarray, y: np.ndarray) -> float:
    preds = np.argmax(logits, axis=1)
    return float(np.mean(preds == y))


def evaluate(model, X, y, batch_size=256):
    criterion = SoftmaxCrossEntropyLoss()

    all_losses = []
    all_preds = []
    all_labels = []

    for batch_x, batch_y in get_batches(X, y, batch_size=batch_size, shuffle=False):
        logits = model.forward(batch_x)
        loss = criterion.forward(logits, batch_y)

        preds = np.argmax(logits, axis=1)

        all_losses.append(loss)
        all_preds.append(preds)
        all_labels.append(batch_y)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    avg_loss = float(np.mean(all_losses))
    acc = float(np.mean(y_pred == y_true))
    return avg_loss, acc


def train_one_epoch(model, criterion, optimizer, X_train, y_train, batch_size=64):
    batch_losses = []
    all_preds = []
    all_labels = []

    for batch_x, batch_y in get_batches(X_train, y_train, batch_size=batch_size, shuffle=True):
        logits = model.forward(batch_x)
        loss = criterion.forward(logits, batch_y)

        optimizer.zero_grad()
        dlogits = criterion.backward()
        model.backward(dlogits)
        optimizer.step()

        preds = np.argmax(logits, axis=1)

        batch_losses.append(loss)
        all_preds.append(preds)
        all_labels.append(batch_y)

    y_pred = np.concatenate(all_preds, axis=0)
    y_true = np.concatenate(all_labels, axis=0)

    train_loss = float(np.mean(batch_losses))
    train_acc = float(np.mean(y_pred == y_true))
    return train_loss, train_acc


def run_single_experiment(
    X_train,
    y_train,
    X_val,
    y_val,
    config,
    input_dim,
    num_classes=10,
):
    model = MLP(
        input_dim=input_dim,
        hidden_dim1=config["hidden_dim1"],
        hidden_dim2=config["hidden_dim2"],
        num_classes=num_classes,
        activation=config["activation"],
        weight_scale=config["weight_scale"],
    )

    criterion = SoftmaxCrossEntropyLoss()

    optimizer = SGD(
        params=model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        lr_decay=config["lr_decay"],
        decay_type=config["decay_type"],
    )

    best_val_acc = -1.0
    best_epoch = -1

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            X_train=X_train,
            y_train=y_train,
            batch_size=config["batch_size"],
        )

        val_loss, val_acc = evaluate(
            model=model,
            X=X_val,
            y=y_val,
            batch_size=256,
        )

        current_lr = optimizer.get_lr()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

        optimizer.schedule()

    result = {
        "config": config,
        "best_val_acc": float(best_val_acc),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(history["train_loss"][-1]),
        "final_train_acc": float(history["train_acc"][-1]),
        "final_val_loss": float(history["val_loss"][-1]),
        "final_val_acc": float(history["val_acc"][-1]),
        "history": history,
    }
    return result


def sort_results(results):
    return sorted(results, key=lambda x: x["best_val_acc"], reverse=True)


def print_top_results(results, top_k=10):
    print("\n=== Top Hyperparameter Search Results ===")
    for i, r in enumerate(results[:top_k], start=1):
        cfg = r["config"]
        print(
            f"[{i}] best_val_acc={r['best_val_acc']:.4f}, "
            f"best_epoch={r['best_epoch']}, "
            f"hidden=({cfg['hidden_dim1']},{cfg['hidden_dim2']}), "
            f"act={cfg['activation']}, "
            f"lr={cfg['lr']}, "
            f"wd={cfg['weight_decay']}, "
            f"batch_size={cfg['batch_size']}"
        )


def main():
    # =========================
    # 1. 基础设置
    # =========================
    base_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(base_dir, "..", "EuroSAT_RGB"))
    save_dir = os.path.join(base_dir, "outputs", "search")
    os.makedirs(save_dir, exist_ok=True)

    image_size = (32, 32)
    seed = 42
    np.random.seed(seed)

    print("Loading dataset...")
    dataset = EuroSATDataset(
        root_dir=root_dir,
        image_size=image_size,
        flatten=True,
        normalize=True,
    )

    X, y = dataset.load_data()

    splits = train_val_test_split(
        X, y,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        shuffle=True,
        seed=seed,
    )

    X_train, y_train = splits["X_train"], splits["y_train"]
    X_val, y_val = splits["X_val"], splits["y_val"]

    X_train, X_val, _ = standardize_data(X_train, X_val, splits["X_test"])

    input_dim = X_train.shape[1]
    num_classes = len(dataset.class_names)

    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)

    # =========================
    # 2. 定义搜索空间
    # 你可以先缩小范围，跑通后再扩大
    # =========================
    search_space = {
        "hidden_dim1": [128, 256],
        "hidden_dim2": [64, 128],
        "activation": ["relu", "tanh"],
        "lr": [0.01, 0.001],
        "weight_decay": [0.0, 1e-4, 1e-3],
        "batch_size": [64],
        "epochs": [40],
        "weight_scale": [None],
        "lr_decay": [0.95],
        "decay_type": ["multiplicative"],
    }

    keys = list(search_space.keys())
    values = [search_space[k] for k in keys]

    all_configs = []
    for combo in itertools.product(*values):
        config = dict(zip(keys, combo))
        # 可以加一个约束，避免 hidden_dim2 > hidden_dim1
        if config["hidden_dim2"] > config["hidden_dim1"]:
            continue
        all_configs.append(config)

    print(f"Total configs to run: {len(all_configs)}")

    # =========================
    # 3. 逐组实验
    # =========================
    results = []

    for idx, config in enumerate(all_configs, start=1):
        print("\n" + "=" * 80)
        print(f"Running config [{idx}/{len(all_configs)}]")
        print(config)

        result = run_single_experiment(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            config=config,
            input_dim=input_dim,
            num_classes=num_classes,
        )

        print(
            f"Done | best_val_acc={result['best_val_acc']:.4f} "
            f"at epoch {result['best_epoch']}"
        )

        results.append(result)

        # 每次跑完就保存，防止中断丢结果
        temp_results_path = os.path.join(save_dir, "search_results_partial.json")
        with open(temp_results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    # =========================
    # 4. 排序并保存最终结果
    # =========================
    results = sort_results(results)

    final_results_path = os.path.join(save_dir, "search_results.json")
    with open(final_results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

    print_top_results(results, top_k=10)
    print(f"\nFinal results saved to: {final_results_path}")

    # 保存最优配置单独一份，后面 train.py 可直接照着改
    best_config = results[0]["config"]
    best_config_path = os.path.join(save_dir, "best_config.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=4, ensure_ascii=False)

    print(f"Best config saved to: {best_config_path}")
    print("\nBest config:")
    print(best_config)


if __name__ == "__main__":
    main()
