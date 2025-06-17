# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import datetime
import os
import logging
import pickle
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from datasets import load_dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from detector.config import Config
from detector.utils import generate_cache_path
from detector.model.neural_based.joint_features import (
    MLPForAIGTDetection,
    collect_features,
    prepare_data,
)

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Number of visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA logical device {i} -> {torch.cuda.get_device_name(i)}")


def train(model, train_dataloader, dev_dataloader, device="cpu"):
    logging.info("Training ...")
    model.to(device)

    epochs = 10
    lr = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    best_accuracy = 0
    train_record = {
        "train_loss": [],
        "train_acc": [],
        "dev_loss": [],
        "dev_acc": [],
    }
    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x_batch, y_batch in train_dataloader:
            optimizer.zero_grad()

            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            logits = model(x_batch)

            loss = loss_fn(logits, y_batch)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct / total

        logging.info(
            "Epoch %d: Loss = %.4f, Accuracy = %.4f", epoch + 1, avg_loss, accuracy
        )

        dev_accuracy = evaluate(model, dev_dataloader, loss_fn, device)
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
            torch.save(
                model.state_dict(),
                os.path.join(
                    Config.CKPT_DIR,
                    f"joint-features-mlp_{timestamp}_{int(dev_accuracy*1000)}.pth",
                ),
            )
            logging.info("Best model saved.")


def evaluate(model, dev_loader, loss_fn, device="cpu"):
    logging.info("Evaluating ...")
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in dev_loader:
            x_batch = x_batch.to(device)
            logits = model(x_batch)
            pred = torch.argmax(logits, dim=1).cpu()
            preds.extend(pred.tolist())
            targets.extend(y_batch.tolist())
    acc = accuracy_score(targets, preds)
    print(f"Validation Accuracy: {acc:.4f}")
    return acc


if __name__ == "__main__":

    # trainset_path = os.path.join(Config.RAW_DATA["train"])
    # devset_path = os.path.join(Config.RAW_DATA["dev"])
    trainset_path = os.path.join(
        Config.DATA_DIR, "data_augment/train_extreme_short.json"
    )
    devset_path = os.path.join(Config.DATA_DIR, "data_augment/dev_extreme_short.json")
    logging.info("Loading train-set from %s ...", repr(trainset_path))
    logging.info("Loading dev-set from %s ...", repr(devset_path))
    dataset_dict = load_dataset(
        "json", data_files={"train": trainset_path, "dev": devset_path}
    )
    print(dataset_dict)

    # 准备训练集数据
    train_extra_kwargs = {
        "labels": dataset_dict["train"]["label"],
        "refer_dataset": dataset_dict["train"],
    }
    train_models_cfg = Config.prepare_models_config(**train_extra_kwargs)
    train_dataset, train_feat_dim = prepare_data(
        dataset=dataset_dict["train"],
        models_config=train_models_cfg,
        dataset_flag="train",
        labels=dataset_dict["train"]["label"],
    )
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    # 准备验证集数据
    dev_extra_kwargs = {
        "labels": dataset_dict["dev"]["label"],
        "refer_dataset": dataset_dict["train"],
    }
    dev_models_cfg = Config.prepare_models_config(**dev_extra_kwargs)
    dev_dataset, dev_feat_dim = prepare_data(
        dataset=dataset_dict["dev"],
        models_config=dev_models_cfg,
        dataset_flag="dev",
        labels=dataset_dict["dev"]["label"],
    )
    dev_loader = DataLoader(dev_dataset, batch_size=16)

    assert train_feat_dim == dev_feat_dim, "trainset feat dim doesn't align with devset"
    print(dev_feat_dim)

    mlp = MLPForAIGTDetection(
        input_dim=train_feat_dim,
        hidden_dim=512,
        output_dim=2,
        dropout=0.3,
    )

    train(mlp, train_loader, dev_loader, device="cuda")
