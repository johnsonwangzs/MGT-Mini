# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os.path
import logging
import pickle
import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    DataCollatorWithPadding,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from tqdm import tqdm
from detector.config import Config
from detector.utils import generate_cache_path
from detector.model.neural_based.base_neural_model import BaseNeuralModel
from detector.eval.evaluator import PerformanceEvaluator


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class CustomChineseBertForSequenceClassification(BertForSequenceClassification):

    def __init__(self, config):
        super().__init__(config)
        # 替换原有的classifier层
        self.classifier = nn.Sequential(
            nn.Linear(
                in_features=self.bert.config.hidden_size,
                out_features=config.classifier_hidden_size,
            ),
            nn.ReLU(),
            nn.Dropout(p=config.classifier_hidden_dropout),
            nn.Linear(
                in_features=config.classifier_hidden_size,
                out_features=config.num_labels,
            ),
        )

    def forward(self, input_ids, attention_mask=None, **kwargs):
        outputs = self.bert(input_ids, attention_mask=attention_mask, **kwargs)
        pooled_output = outputs.pooler_output  # 使用 Bert 提供的池化输出
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ChineseBertForAIGTDetection(BaseNeuralModel):

    desc = repr("Supervise fine-tuned Roberta for AIGT detection")

    def __init__(self, model_path: str, tokenizer_path: str):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path

    def _load_model(self):
        logging.info("Loading RoBERTa SFT model: %s", self.model_path)
        self.model = CustomChineseBertForSequenceClassification.from_pretrained(
            self.model_path
        )
        # print(self.model)
        self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)

    def __call__(self, text: str, **kwargs):
        return self.inference_example(text, kwargs)

    def _tokenize_fn(self, example):
        tokenized = self.tokenizer(
            example["text"], padding="longest", truncation=True, max_length=512
        )
        if "label" in example.keys():
            tokenized["labels"] = example["label"]  # 这里将 `label` 改为 `labels`
        return tokenized

    def _process_dataloader(self, dataset: Dataset):
        # 对数据集中的所有样本做tokenize，确保移除无用的原始列，如 `text`，避免 batch 仍然包含字符串
        logging.info("Tokenizing the dataset ...")
        dataset_columns = dataset.column_names
        if set(dataset_columns) == set(["text", "label", "model", "source"]):
            tokenized_datasets = dataset.map(
                self._tokenize_fn,
                batched=True,
                remove_columns=["text", "label", "model", "source"],
            )
        elif set(dataset_columns) == set(["text", "label"]):
            tokenized_datasets = dataset.map(
                self._tokenize_fn,
                batched=True,
                remove_columns=["text", "label"],
            )
        elif set(dataset_columns) == set(["text", "id"]):
            tokenized_datasets = dataset.map(
                self._tokenize_fn,
                batched=True,
                remove_columns=["text", "id"],
            )
        else:
            raise AssertionError("Unseen dataset columns")

        # print(tokenized_datasets)

        data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer, return_tensors="pt"
        )
        dataloader = DataLoader(
            tokenized_datasets,
            batch_size=8,
            collate_fn=data_collator,
            num_workers=4,
        )

        return dataloader

    def _batch_inference(self, dataloader, device):

        self.model.eval()
        total_preds = []
        total_logits = []
        total_cls_vec = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inferring ..."):
                input_ids = batch["input_ids"].clone().detach().to(device)
                attention_mask = batch["attention_mask"].clone().detach().to(device)

                outputs = self.model(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True
                )

                logits = outputs.logits
                total_logits += logits.cpu().numpy().tolist()

                batch_preds = torch.argmax(logits, dim=1)
                total_preds += batch_preds.cpu().numpy().tolist()

                cls_vec = outputs.hidden_states[-1][:, 0, :]
                total_cls_vec += cls_vec.cpu().numpy().tolist()

        return total_preds, total_logits, total_cls_vec

    def inference_example(self, text: str, device="cpu", **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        logging.info("Using device: %s", device)
        self._load_model()
        self.model.to(device)
        inputs = self.tokenizer(text, return_tensors="pt").to(device)
        outputs = self.model(**inputs, output_hidden_states=True)
        pred = torch.argmax(outputs.logits)
        print(f"Text: {repr(text)}")
        print(f"Logits: {outputs.logits.cpu().detach().numpy().tolist()}")
        print(f"Prediction: {pred.item()}")

    def inference_dataset(self, dataset: Dataset, device="cpu", **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)

        preds_cache_path = generate_cache_path(
            data=dataset,
            filename=f"{self.__class__.__name__}.tmp",
            model_name=self.model_path,
        )
        if not os.path.exists(preds_cache_path):
            # 执行推理
            logging.info("Using device: %s", device)
            self._load_model()
            dataloader = self._process_dataloader(dataset)
            self.model.to(device)
            preds, logits, cls_vec = self._batch_inference(dataloader, device)

            cache_data = {"preds": preds, "logits": logits, "cls_vec": cls_vec}
            with open(preds_cache_path, "wb") as f_cache:
                pickle.dump(cache_data, f_cache)
                logging.info(
                    "Save model prediction results to cache `%s`", preds_cache_path
                )
        else:
            # 直接读取缓存推理结果
            logging.info("Reading prediction results from cache `%s`", preds_cache_path)
            with open(preds_cache_path, "rb") as f_cache:
                cache_data = pickle.load(f_cache)

        return cache_data


if __name__ == "__main__":

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    logging.info("Loading dataset ...")
    devset_path = os.path.join(Config.RAW_DATA["dev"])
    dataset_dict = load_dataset("json", data_files={"dev": devset_path})
    print(dataset_dict)
    devset = dataset_dict["dev"]

    roberta_sft_path = os.path.join(
        Config.CKPT_DIR, "chinese-roberta-sft_03-21_16-39_938"
    )
    # roberta_sft_path = os.path.join(Config.CKPT_DIR, "chinese-roberta-sft_03-21_14-11_926")
    base_model_path = os.path.join(Config.CKPT_DIR, "chinese-roberta-wwm-ext-large")
    roberta_sft = ChineseBertForAIGTDetection(roberta_sft_path, base_model_path)

    results = roberta_sft.inference_dataset(devset, device)
    evaluator = PerformanceEvaluator()
    labels = dataset_dict["dev"]["label"]
    evaluator.calculate_classification_performance(labels, results["preds"])

    # 测试单样本推理
    # test_human = "【地址】世贸广场7楼，钟表后面的电梯可以直达。交通便利，地理位置好找。\n\n【环境】这家秀玉算是比较大的一家，可以选择自己喜欢的座位，环境不错，很适合聚会聊天。\n\n【服务】服务很热情，从一进门开始就有人主动询问，一路有人把你带到座位上。整个用餐过程的服务都做到及时，主动，热情，挺好的。\n\n【口味】连锁餐厅，口味有一定保障。虽然没有什么创新菜式，但是用餐，聚会，下午茶什么的没问题。老公喜欢这里的牛排，因为柔嫩分量足，吃得很饱；我喜欢这里的煲仔饭和泰皇炒饭，还有水果沙拉，吐司。吃完饭如果不想接着逛街，可以休息会，2点就有下午茶了，点上一壶茶，边喝边聊。唯一缺点就是世贸禁止明火，茶下面不能有蜡烛保温，这个天气，茶一会就冷了。但是我觉得可以用个插电的保温垫，服务是需要更细致，用心的！"
    # roberta_sft.inference_example(test_human, device="cpu")

    # 测试自定义模型类
    # base_model_path = os.path.join(Config.CKPT_DIR, "chinese-roberta-wwm-ext-large")
    # model_config = BertConfig.from_pretrained(base_model_path)
    # model_config.classifier_dropout = 0.3
    # model_config.classifier_hidden_size = 512
    # model_config.classifier_hidden_dropout = 0.3
    # model_config.num_labels = 2
    # print(model_config)
    # model = CustomChineseBertForSequenceClassification(model_config)
    # print(model)
