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


class CommonPhraseModel(BaseRuleModel):
    """对于待测文本，检测是否存在常见的机器短语"""

    desc = repr(
        "The machine regularly generates some representative phrases, such as `总的来说`, `总而言之`, etc."
    )

    def __init__(self):
        super().__init__()
        self.rule_patterns = [
            r"总的来说，",
            r"总体来说，",
            r"总体而言，",
            r"总而言之，",
            r"总之，",
            r"总是",
        ]
        self.combined_patterns = "|".join(self.rule_patterns)

    def inference_dataset(self, dataset: Dataset, **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        preds, feats = [], []
        for example in dataset["text"]:
            pattern_cnt = len(re.findall(self.combined_patterns, example))
            feats.append(pattern_cnt)
            preds.append(1 if pattern_cnt > 0 else 0)

        return {"preds": preds, "pattern_cnt": feats}

    def inference_example(self, text, **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        return 1 if len(re.findall(self.combined_patterns, text)) > 0 else 0

    def __call__(self, text):
        return self.inference_example(text)


if __name__ == "__main__":

    logging.info("Loading dataset ...")
    devset_path = os.path.join(Config.RAW_DATA["dev"])
    dataset_dict = load_dataset("json", data_files={"dev": devset_path})
    devset = dataset_dict["dev"]

    model = CommonPhraseModel()
    results = model.inference_dataset(devset)

    evaluator = PerformanceEvaluator()
    labels = dataset_dict["dev"]["label"]
    evaluator.calculate_classification_performance(labels, results["preds"])
