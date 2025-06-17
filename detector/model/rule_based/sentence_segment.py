# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os
import re
import logging
from datasets import load_dataset, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from detector.config import Config
from detector.eval.evaluator import PerformanceEvaluator
from detector.model.rule_based.base_rule_model import BaseRuleModel


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class SentenceSegmentModel(BaseRuleModel):
    """对于待测文本，分析连续子句（逗号间隔视为子句，句号视为结束）的数量分布"""

    desc = repr("Sentence segmentation. Use comma as delimiter.")

    def __init__(self, decision_threshold: float = 10.0, **kwargs):
        """_summary_

        Args:
            threshold_consecutive_commas (int): 连续加权分数判定阈值
        """
        super().__init__()
        self.sep_punctuation = ["，", "。"]
        self.rule_pattern = {
            "single_period": r"[^。]。[^。]",
            "single_comma": r"[^，]，[^，]",
        }
        self.decision_threshold = decision_threshold
        self.weight_coefficient = {
            1: 0.0,
            2: 1.0,
            3: 2.0,
            4: 4.0,
            5: 8.0,
            6: 16.0,
            7: 32.0,
        }

    def _extract_feature(self, text: str):
        """从一段文本中提取特征（考虑文本长度归一化）"""
        # 分别统计逗号和句号的数量
        pattern_cnt = {
            "single_period": len(re.findall(self.rule_pattern["single_period"], text)),
            "single_comma": len(re.findall(self.rule_pattern["single_comma"], text)),
        }
        num_total = sum(pattern_cnt.values())

        # 统计连续逗号（直到遇到句号为止）
        commas_per_period = []
        num_consecutive_commas = 0
        for ch in text:
            if ch == "，":
                num_consecutive_commas += 1
            if ch in ["。", "！", "？", "；"]:
                commas_per_period.append(num_consecutive_commas)
                num_consecutive_commas = 0

        # 计算加权总分（连续逗号越多越高）
        max_key = max(self.weight_coefficient.keys())
        weighted_comma_score = sum(
            [
                n * self.weight_coefficient.get(n, self.weight_coefficient[max_key])
                for n in commas_per_period
            ]
        )

        # 文本归一化（句子数归一化）
        num_sentences = sum(text.count(p) for p in ["。", "！", "？", "；"])
        normalized_score = weighted_comma_score / max(num_sentences, 1)

        feature = []
        feature.append(normalized_score)
        feature.append(pattern_cnt["single_period"] / max(num_total, 1))
        feature.append(pattern_cnt["single_comma"] / max(num_total, 1))
        feature.append(
            pattern_cnt["single_comma"] / max(pattern_cnt["single_period"], 1)
        )

        return feature

    def inference_dataset(self, dataset: Dataset, **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        preds, feats = [], []
        for example in dataset["text"]:
            feature = self._extract_feature(example)
            preds.append(int(feature[0] < self.decision_threshold))
            feats.append(feature)
        return {"preds": preds, "feats": feats}

    def inference_example(self, text: str, **kwargs):
        return super().inference_example(text, **kwargs)

    def draw_plot(self, dataset, labels):
        evaluator = PerformanceEvaluator()  # 性能评估
        thresholds = []
        f1s, precisions, recalls = [], [], []
        for threshold in tqdm(range(1, 30), desc="Traversing thresholds ..."):
            self.decision_threshold = threshold
            results = self.inference_dataset(dataset)
            thresholds.append(threshold)
            precision, recall, f1 = evaluator.calculate_classification_performance(
                labels, results["preds"], print_result=False
            )
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        plt.figure(figsize=(8, 5))
        plt.plot(thresholds, f1s, marker="o", linestyle="-", label="f1")
        plt.plot(thresholds, precisions, marker="o", linestyle="-", label="precision")
        plt.plot(thresholds, recalls, marker="o", linestyle="-", label="recall")
        plt.xlabel("threshold")
        plt.ylabel("metric")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.__class__.__name__}.png", dpi=300, bbox_inches="tight")
        plt.close()
        # plt.show()


if __name__ == "__main__":

    trainset_path = os.path.join(Config.RAW_DATA["train"])
    devset_path = os.path.join(Config.RAW_DATA["dev"])
    logging.info("Loading train-set from %s ...", repr(trainset_path))
    logging.info("Loading dev-set from %s ...", repr(devset_path))
    dataset_dict = load_dataset(
        "json", data_files={"train": trainset_path, "dev": devset_path}
    )
    print(dataset_dict)

    sentence_segment_model = SentenceSegmentModel()
    sentence_segment_model.draw_plot(
        dataset_dict["train"], dataset_dict["train"]["label"]
    )
