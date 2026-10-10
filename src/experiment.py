
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import torch
from torch import nn
import mlflow
import mlflow.pytorch
import numpy as np
import optuna
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT = "fashion-mnist-cnn"
MODEL_NAME = "fashion-mnist-cnn-classifier"
DATA_ROOT = Path("data")
RAW_ROOT = DATA_ROOT / "FashionMNIST"
ARTIFACT_DIR = Path("outputs")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

N_TRAIN = int(os.getenv("FASHION_TRAIN_LIMIT", "6000"))
N_TEST = int(os.getenv("FASHION_TEST_LIMIT", "1000"))
EPOCHS_TUNE = int(os.getenv("TUNE_EPOCHS", "2"))
EPOCHS_FINAL = int(os.getenv("FINAL_EPOCHS", "4"))

CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_hash():
    """Hash SHA-256 dataset untuk melacak versi data pada setiap run."""
    h = hashlib.sha256()
    files = sorted(p for p in RAW_ROOT.rglob("*") if p.is_file())

    if not files:
        raise FileNotFoundError(
            "Dataset belum ada. Jalankan: python src/prepare_data.py"
        )

    for p in files:
        h.update(str(p.relative_to(RAW_ROOT)).encode())
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)

    return h.hexdigest()


def make_datasets(augment=False):
    train_tf = (
        [transforms.RandomHorizontalFlip(p=0.5)]
        if augment else []
    )
    train_tf += [
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ]

    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_full = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=transforms.Compose(train_tf)
    )

    eval_full = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=eval_tf
    )

    n = min(N_TRAIN, len(train_full))
    indices = np.arange(n)

    tr_idx, val_idx = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=np.array(train_full.targets[:n])
    )

    train_ds = Subset(train_full, tr_idx.tolist())
    val_ds = Subset(eval_full, val_idx.tolist())

    test_ds = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=eval_tf
    )
    test_ds = Subset(
        test_ds,
        list(range(min(N_TEST, len(test_ds))))
    )

    return train_ds, val_ds, test_ds


class SmallCNN(nn.Module):
    def __init__(self, channels=16, dropout=0.25):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(channels, channels * 2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.classifier = nn.Sequential(
            nn.Linear(channels * 2 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def evaluate(model, loader):
    model.eval()
    ys, preds, losses = [], [], []
    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            logits = model(x)
            losses.append(loss_fn(logits, y).item())
            preds.extend(logits.argmax(1).cpu().numpy().tolist())
            ys.extend(y.cpu().numpy().tolist())

    return (
        float(np.mean(losses)),
        accuracy_score(ys, preds),
        f1_score(ys, preds, average="macro")
    )


def train_one_run(
    run_name,
    params,
    seed=42,
    epochs=EPOCHS_TUNE,
    augment=True,
    use_dropout=True,
    nested=False,
    log_model=True
):
    set_seed(seed)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    dhash = dataset_hash()
    train_ds, val_ds, _ = make_datasets(augment=augment)

    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=int(params["batch_size"]),
        shuffle=True,
        num_workers=0,
        generator=generator
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=0
    )

    dropout = float(params["dropout"]) if use_dropout else 0.0
    model = SmallCNN(
        channels=int(params["channels"]),
        dropout=dropout
    ).to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(params["lr"])
    )
    loss_fn = nn.CrossEntropyLoss()
    start = time.time()

    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.set_tags({
            "dataset": "Fashion-MNIST-subset",
            "device": DEVICE,
            "stage": run_name.split("-")[0],
            "framework": "pytorch"
        })

        logged_params = dict(params)
        logged_params.update({
            "seed": seed,
            "epochs": epochs,
            "augment": augment,
            "use_dropout": use_dropout,
            "train_limit": N_TRAIN,
            "test_limit": N_TEST,
            "device": DEVICE
        })

        mlflow.log_params(logged_params)
        mlflow.log_param("dataset_sha256", dhash)
        mlflow.log_param("validation_split_seed", 42)
        mlflow.log_param("torch_version", torch.__version__)
        mlflow.log_param("python_device", DEVICE)

        best_acc = -1.0
        best_state = None

        for epoch in range(epochs):
            model.train()
            train_losses = []

            for x, y in train_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_loss, val_acc, val_f1 = evaluate(
                model, val_loader
            )

            mlflow.log_metrics({
                "train_loss": float(np.mean(train_losses)),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_f1_macro": val_f1
            }, step=epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()
                }

        if best_state:
            model.load_state_dict(best_state)

        mlflow.log_metric("best_val_accuracy", best_acc)
        mlflow.log_metric(
            "duration_seconds", time.time() - start
        )


        if log_model:
            example_input = torch.randn(1, 1, 28, 28)

            mlflow.pytorch.log_model(
                pytorch_model=model.cpu(),
                artifact_path="model",
                input_example=example_input.numpy(),
                serialization_format="pickle"
            )

            model.to(DEVICE)


    return best_acc, model


def default_params():
    return {
        "lr": 0.001,
        "batch_size": 64,
        "channels": 16,
        "dropout": 0.25
    }


def baseline(args):
    train_one_run(
        "baseline-cnn",
        default_params(),
        seed=42,
        epochs=args.epochs,
        augment=True,
        use_dropout=True
    )


def tune(args):
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    with mlflow.start_run(run_name="optuna-sweep-20-trials"):
        mlflow.log_param("requested_trials", args.trials)

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42)
        )

        def objective(trial):
            params = {
                "lr": trial.suggest_float(
                    "lr", 1e-4, 3e-3, log=True
                ),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [32, 64, 128]
                ),
                "channels": trial.suggest_categorical(
                    "channels", [8, 16, 24]
                ),
                "dropout": trial.suggest_float(
                    "dropout", 0.0, 0.5
                )
            }

            score, _ = train_one_run(
                f"trial-{trial.number:02d}",
                params,
                seed=42,
                epochs=args.epochs,
                augment=True,
                use_dropout=True,
                nested=True,
                log_model=False
            )
            trial.set_user_attr(
                "mlflow_validation_accuracy", score
            )
            return score

        study.optimize(objective, n_trials=args.trials)

        mlflow.log_params({
            f"best_{k}": v
            for k, v in study.best_params.items()
        })
        mlflow.log_metric(
            "best_val_accuracy", study.best_value
        )

        ARTIFACT_DIR.mkdir(exist_ok=True)
        result = {
            "best_params": study.best_params,
            "best_val_accuracy": study.best_value,
            "trials": len(study.trials)
        }

        (ARTIFACT_DIR / "best_params.json").write_text(
            json.dumps(result, indent=2)
        )
        mlflow.log_artifact(
            str(ARTIFACT_DIR / "best_params.json")
        )
        print(json.dumps(result, indent=2))


def load_best_params():
    path = ARTIFACT_DIR / "best_params.json"

    if path.exists():
        return json.loads(path.read_text())["best_params"]

    print("best_params.json belum ada; memakai konfigurasi default.")
    return default_params()


def seeds(args):
    params = load_best_params()
    scores = []

    for seed in [42, 123, 2026]:
        score, _ = train_one_run(
            f"best-config-seed-{seed}",
            params,
            seed=seed,
            epochs=args.epochs,
            augment=True,
            use_dropout=True
        )
        scores.append(score)

    mean = float(np.mean(scores))
    std = float(np.std(scores, ddof=1))

    summary = {
        "seeds": [42, 123, 2026],
        "val_accuracy": scores,
        "mean": mean,
        "std_sample": std
    }

    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "seed_summary.json").write_text(
        json.dumps(summary, indent=2)
    )

    print("Validation accuracy per seed:", scores)
    print(f"Mean ± sample std: {mean:.4f} ± {std:.4f}")


def ablation(args):
    params = load_best_params()
    variants = [
        ("ablation-full", True, True),
        ("ablation-no-augmentation", False, True),
        ("ablation-no-dropout", True, False)
    ]

    results = []

    for name, augment, dropout in variants:
        score, _ = train_one_run(
            name,
            params,
            seed=42,
            epochs=args.epochs,
            augment=augment,
            use_dropout=dropout
        )
        results.append({
            "run": name,
            "augment": augment,
            "dropout": dropout,
            "best_val_accuracy": score
        })

    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "ablation_results.json").write_text(
        json.dumps(results, indent=2)
    )
    print(json.dumps(results, indent=2))


def register(args):
    params = load_best_params()

    train_one_run(
        "champion-final-model",
        params,
        seed=42,
        epochs=args.epochs,
        augment=True,
        use_dropout=True
    )

    client = mlflow.tracking.MlflowClient(
        tracking_uri=TRACKING_URI
    )
    exp = client.get_experiment_by_name(EXPERIMENT)

    runs = client.search_runs(
        [exp.experiment_id],
        order_by=["attributes.start_time DESC"],
        max_results=50
    )

    target = next(
        (
            run for run in runs
            if run.data.tags.get("mlflow.runName")
            == "champion-final-model"
        ),
        None
    )

    if target is None:
        raise RuntimeError(
            "Run champion-final-model tidak ditemukan."
        )

    model_uri = f"runs:/{target.info.run_id}/model"
    version = mlflow.register_model(model_uri, MODEL_NAME)

    client.set_registered_model_alias(
        MODEL_NAME, "champion", version.version
    )

    print(
        f"Registered {MODEL_NAME} version "
        f"{version.version} with alias @champion"
    )


def predict(args):
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@champion"

    model = mlflow.pytorch.load_model(
        model_uri,
        map_location=DEVICE
    )

    # Model pt2 harus dikonversi ke modul PyTorch.
    if isinstance(model, torch.export.ExportedProgram):
        model = model.module()

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"Model bukan modul PyTorch biasa: {type(model)}"
        )

    model = model.to(DEVICE)
    model.eval()

    _, _, test_ds = make_datasets(augment=False)
    loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    x, y = next(iter(loader))
    x = x.to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        pred = int(logits.argmax(dim=1).item())

    actual = int(y.item())

    print(f"Model URI: {model_uri}")
    print(f"Prediksi kelas: {pred} ({CLASSES[pred]})")
    print(f"Label aktual: {actual} ({CLASSES[actual]})")


def main():
    parser = argparse.ArgumentParser(
        description="Fashion-MNIST MLOps experiments"
    )
    sub = parser.add_subparsers(
        dest="command", required=True
    )

    for name in [
        "baseline", "tune", "seeds",
        "ablation", "register", "predict"
    ]:
        p = sub.add_parser(name)
        default_epochs = (
            EPOCHS_TUNE
            if name in ["baseline", "tune"]
            else EPOCHS_FINAL
        )
        p.add_argument(
            "--epochs", type=int, default=default_epochs
        )

        if name == "tune":
            p.add_argument(
                "--trials", type=int, default=20
            )

    args = parser.parse_args()

    commands = {
        "baseline": baseline,
        "tune": tune,
        "seeds": seeds,
        "ablation": ablation,
        "register": register,
        "predict": predict
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
