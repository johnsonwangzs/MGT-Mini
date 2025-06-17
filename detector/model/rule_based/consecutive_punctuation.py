# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os.path
import re
import logging
from datasets import load_dataset, Dataset
from detector.config import Config
from detector.model.rule_based.base_rule_model import BaseRuleModel
from detector.eval.evaluator import PerformanceEvaluator


class ConsecutivePunctuationModel(BaseRuleModel):
    """对于待测文本，检测其中是否存在连续的标点符号"""

    desc = repr(
        "Consecutive punctuation are considered as human text, such as `。。。`, `！！！`, etc."
    )

    def __init__(self):
        super().__init__()
        self.rule_patterns = [
            r"！！！",
            r"。。。",
        ]
        self.escaped_patterns = [
            re.escape(p) for p in self.rule_patterns
        ]  # 自动转义特殊字符
        self.combined_patterns = "|".join(self.escaped_patterns)  # 生成安全的正则表达式

    def inference_dataset(self, dataset: Dataset, **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        preds, feats = [], []
        for example in dataset["text"]:
            pattern_cnt = len(re.findall(self.combined_patterns, example))
            feats.append(pattern_cnt)
            preds.append(0 if pattern_cnt > 0 else 1)

        return {"preds": preds, "pattern_cnt": feats}

    def inference_example(self, text, **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        return 0 if len(re.findall(self.combined_patterns, text)) > 0 else 1

    def __call__(self, text):
        return self.inference_example(text)


if __name__ == "__main__":

    logging.info("Loading dataset ...")
    devset_path = os.path.join(Config.RAW_DATA["dev"])
    dataset_dict = load_dataset("json", data_files={"dev": devset_path})
    devset = dataset_dict["dev"]

    model = ConsecutivePunctuationModel()
    results = model.inference_dataset(devset)

    evaluator = PerformanceEvaluator()
    labels = dataset_dict["dev"]["label"]
    evaluator.calculate_classification_performance(labels, results["preds"])
