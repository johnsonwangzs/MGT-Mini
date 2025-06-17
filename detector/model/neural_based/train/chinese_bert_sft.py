# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os
import datetime
import json
import logging
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
from transformers import (
    BertConfig,
    BertTokenizer,
    get_scheduler,
    DataCollatorWithPadding,
)
from detector.config import Config
from detector.model.neural_based.chinese_bert_detector import (
    CustomChineseBertForSequenceClassification,
)

torch.manual_seed(1618)
torch.cuda.manual_seed_all(1618)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def eval_model(model, dev_dataloader, loss_fn, device):

    model.eval()
    correct = 0
    num_examples = 0
    total_loss = 0.0
    total_predictions = []
    with torch.no_grad():
        for batch in tqdm(dev_dataloader, desc="Evaluating..."):
            input_ids = batch["input_ids"].clone().detach().to(device)
            attention_mask = batch["attention_mask"].clone().detach().to(device)
            labels = batch["labels"].clone().detach().to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            total_loss += loss.detach().item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total_predictions += predictions.cpu().numpy().tolist()
            num_examples += labels.size(0)

    accuracy = correct / num_examples
    avg_loss = total_loss / len(dev_dataloader)

    return accuracy, total_predictions, avg_loss


def freeze_bert_base(model):

    for param in model.bert.parameters():
        param.requires_grad = False


def train_model(model, train_dataloader, dev_dataloader, optimizer, loss_fn, device):

    # 设置学习率调度器
    epochs = 20
    num_training_steps = len(train_dataloader) * epochs
    num_warmup_steps = int(0.05 * num_training_steps)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    # 训练循环
    best_accuracy = 0
    train_record = {
        "train_loss": [],
        "train_acc": [],
        "dev_loss": [],
        "dev_acc": [],
    }
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in tqdm(
            train_dataloader,
            desc=f"Training Epoch {epoch+1}...",
        ):
            optimizer.zero_grad()

            # 获取输入数据
            input_ids = batch["input_ids"].clone().to(device)
            attention_mask = batch["attention_mask"].clone().to(device)
            labels = batch["labels"].clone().to(device)

            # 前向传播
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # 计算损失
            loss = loss_fn(logits, labels)
            total_loss += loss.detach().item()

            # 反向传播
            loss.backward()
            lr_scheduler.step()
            optimizer.step()

            # 计算准确率
            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_dataloader)
        accuracy = correct / total

        logging.info(
            "Epoch %d: Loss = %.4f, Accuracy = %.4f", epoch + 1, avg_loss, accuracy
        )

        # 验证
        dev_accuracy, _, dev_loss = eval_model(model, dev_dataloader, loss_fn, device)
        logging.info("Accuracy on devset: %.4f", dev_accuracy)

        # 保存最佳模型
        if dev_accuracy > best_accuracy:
            best_accuracy = dev_accuracy
            timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
            model.save_pretrained(
                os.path.join(
                    Config.CKPT_DIR,
                    f"chinese-bert-sft_{timestamp}_{int(dev_accuracy*1000)}",
                )
            )
            logging.info("Best model saved.")

        train_record["train_acc"].append(accuracy)
        train_record["train_loss"].append(avg_loss)
        train_record["dev_acc"].append(dev_accuracy)
        train_record["dev_loss"].append(dev_loss)

    return train_record


if __name__ == "__main__":

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # base_model_path = os.path.join(Config.CKPT_DIR, "chinese-bert-wwm-ext")
    base_model_path = os.path.join(Config.CKPT_DIR, "chinese-roberta-wwm-ext-large")
    logging.info("Loading bert base model (%s) ...", base_model_path)
    tokenizer = BertTokenizer.from_pretrained(base_model_path)

    model_config = BertConfig.from_pretrained(base_model_path)
    model_config.classifier_dropout = 0.3
    model_config.classifier_hidden_size = 512
    model_config.classifier_hidden_dropout = 0.3
    model_config.num_labels = 2
    print(model_config)

    model = CustomChineseBertForSequenceClassification.from_pretrained(
        base_model_path, config=model_config
    )
    print(model)

    logging.info("Loading trainset and devset ...")
    # trainset_path = os.path.join(Config.RAW_DATA["train"])
    # devset_path = os.path.join(Config.RAW_DATA["dev"])
    trainset_path = os.path.join(
        Config.DATA_DIR, "data_augment/train_extreme_short.json"
    )
    devset_path = os.path.join(Config.DATA_DIR, "data_augment/dev_extreme_short.json")
    dataset = load_dataset(
        "json", data_files={"train": trainset_path, "dev": devset_path}
    )
    print(dataset)

    def tokenize_fn(example):
        tokenized = tokenizer(
            example["text"], padding="longest", truncation=True, max_length=512
        )
        tokenized["labels"] = example["label"]  # 这里将 `label` 改为 `labels`
        return tokenized

    # 对数据集中的所有样本做tokenize，确保移除无用的原始列，如 `text`，避免 batch 仍然包含字符串
    logging.info("Tokenizing the dataset ...")
    tokenized_datasets = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text", "label"],
        # tokenize_fn, batched=True, remove_columns=["text", "model", "source", "label"]
    )
    print(tokenized_datasets)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    train_dataloader = DataLoader(
        tokenized_datasets["train"],
        batch_size=8,
        shuffle=True,
        collate_fn=data_collator,
    )
    dev_dataloader = DataLoader(
        tokenized_datasets["dev"], batch_size=8, collate_fn=data_collator, num_workers=4
    )

    # Debug batch 数据格式
    # batch = next(iter(train_dataloader))
    # for key, value in batch.items():
    #     print(
    #         f"{key}: {type(value)}, shape: {value.shape if isinstance(value, torch.Tensor) else 'not tensor'}"
    #     )

    # 冻结主干（不建议）
    # freeze_bert_base(model)

    model.to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    record = train_model(
        model, train_dataloader, dev_dataloader, optimizer, loss_fn, device
    )
    print(record)

    timestamp = datetime.datetime.now().strftime("%m-%d_%H-%M")
    log_file = os.path.join(Config.LOG_DIR, f"chinese-bert-sft_{timestamp}.log")
    with open(log_file, "w", encoding="utf-8") as f_log:
        json.dump(record, f_log, indent=4, ensure_ascii=False)
