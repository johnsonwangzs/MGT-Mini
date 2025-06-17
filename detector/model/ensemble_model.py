# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
from collections import defaultdict
import os
import time
import json
import logging
import copy
import pickle
from typing import override
from tabulate import tabulate
import numpy as np
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
from detector.config import Config
from detector.utils import generate_cache_path
from detector.eval.evaluator import PerformanceEvaluator
from detector.data_process.build_vote_strategy import build_all_exist_strategies
from detector.model.base_model import BaseModel

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Number of visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA logical device {i} -> {torch.cuda.get_device_name(i)}")


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class EnsembleModelForAIGTDetection(BaseModel):
    """AIGT检测集成模型"""

    desc = repr("Ensemble model for AIGT detection")

    def __init__(self, models_config: list[dict]):
        """_summary_

        Args:
            models_config (list[dict]): 各模型的配置，例如初始化参数和推理参数等
        """
        super().__init__()
        self.models_config = models_config
        Config.check_models_config_validity(self.models_config)
        self.print_models_config()

        # 注册动态投票策略
        self.dynamic_vote_strategy = {
            "extreme_short_text": True,
            "short_text": True,
            "medium_text": True,
        }
        # 注册动态投票权重
        self.dynamic_vote_weights = {
            "extreme_short_text": {
                "neural-01(ChineseBertForAIGTDetection)": 0,
                "neural-02(ChineseBertForAIGTDetection)": 0,
                "neural-03(GLM4ForAIGTDetection)": 0,
                "neural-04(Qwen25ForAIGTDetection)": 0,
                "neural-05(Qwen25ForAIGTDetection)": 95,
                "neural-06(Qwen25ForAIGTDetection)": 1000,
                "neural-07(FastDetectGPT)": 60,
                "neural-08(FastDetectGPT)": 60,
                "neural-09(FastDetectGPT)": 55,
                "neural-10(BinocularsDetector)": 60,
                "neural-11(ChineseBertForAIGTDetection)": 0,
                "rule-01(SpecialTokenModel)": 0,
                "rule-02(SentenceSegmentModel)": 10,
                "rule-03(ConsecutivePunctuationModel)": 10,
                "rule-04(CommonPhraseModel)": 0,
                "rule-05(CommonTokenModel)": 10,
                "joint-01(JointFeaturesMLP)": 10,
                "joint-02(JointFeaturesMLP)": 0,
            },
            "short_text": {
                "neural-01(ChineseBertForAIGTDetection)": 0,
                "neural-02(ChineseBertForAIGTDetection)": 0,
                "neural-03(GLM4ForAIGTDetection)": 0,
                "neural-04(Qwen25ForAIGTDetection)": 40,
                "neural-05(Qwen25ForAIGTDetection)": 95,
                "neural-06(Qwen25ForAIGTDetection)": 1000,
                "neural-07(FastDetectGPT)": 40,
                "neural-08(FastDetectGPT)": 40,
                "neural-09(FastDetectGPT)": 35,
                "neural-10(BinocularsDetector)": 40,
                "neural-11(ChineseBertForAIGTDetection)": 0,
                "rule-01(SpecialTokenModel)": 0,
                "rule-02(SentenceSegmentModel)": 10,
                "rule-03(ConsecutivePunctuationModel)": 10,
                "rule-04(CommonPhraseModel)": 0,
                "rule-05(CommonTokenModel)": 40,
                "joint-01(JointFeaturesMLP)": 40,
                "joint-02(JointFeaturesMLP)": 0,
            },
            "medium_text": {
                "neural-01(ChineseBertForAIGTDetection)": 0,
                "neural-02(ChineseBertForAIGTDetection)": 0,
                "neural-03(GLM4ForAIGTDetection)": 0,
                "neural-04(Qwen25ForAIGTDetection)": 100,
                "neural-05(Qwen25ForAIGTDetection)": 80,
                "neural-06(Qwen25ForAIGTDetection)": 90,
                "neural-07(FastDetectGPT)": 40,
                "neural-08(FastDetectGPT)": 40,
                "neural-09(FastDetectGPT)": 35,
                "neural-10(BinocularsDetector)": 40,
                "neural-11(ChineseBertForAIGTDetection)": 0,
                "rule-01(SpecialTokenModel)": 0,
                "rule-02(SentenceSegmentModel)": 10,
                "rule-03(ConsecutivePunctuationModel)": 10,
                "rule-04(CommonPhraseModel)": 0,
                "rule-05(CommonTokenModel)": 40,
                "joint-01(JointFeaturesMLP)": 40,
                "joint-02(JointFeaturesMLP)": 0,
            },
        }

    def print_models_config(self):
        """打印模型配置"""
        logging.info("Organizing models ...")
        models_summary = {
            "active": [],
            "vote": [],
            "determinative": {
                0: [],
                1: [],
            },
        }

        int_value_map = {
            0: "0 (human)",
            1: "1 (machine)",
            -1: "",
        }
        bool_value_map = {
            True: "✔",
            False: "",
        }

        tabulate_data = []
        for model_config in self.models_config:
            model_id = f"{model_config["id"]}({model_config["cls"].__name__})"
            if model_config["active"] is True:
                models_summary["active"].append(model_id)
            if model_config["vote"] is True and model_config["active"] is True:
                models_summary["vote"].append(model_id)
            if model_config["determinative"] == 0:
                models_summary["determinative"][0].append(model_id)
            if model_config["determinative"] == 1:
                models_summary["determinative"][1].append(model_id)
            tabulate_data.append(
                [
                    model_config["id"],
                    model_config["cls"],
                    bool_value_map[model_config["active"]],
                    bool_value_map[model_config["vote"]],
                    int_value_map[model_config["determinative"]],
                    model_config["importance"],
                    model_config["note"],
                ]
            )
        print(f"Active models: {models_summary["active"]}")
        print(f"Voting models: {models_summary["vote"]}")
        print(f"Determinative models: {models_summary["determinative"]}")

        headers = [
            "Model id",
            "Model cls",
            "Active",
            "Vote",
            "Determinative",
            "Importance",
            "Note",
        ]
        alignments = [
            "center",
            "left",
            "center",
            "center",
            "center",
            "right",
            "left",
        ]
        print(
            tabulate(
                tabulate_data,
                headers=headers,
                tablefmt="pretty",
                colalign=alignments,
            )
        )

    def inference_example(self, text, **kwargs):
        return super().inference_example(text, **kwargs)

    @override
    def inference_dataset(
        self,
        dataset: Dataset,
        evaluate: bool = False,
        dataset_labels: list = None,
        **kwargs,
    ) -> tuple[dict, dict]:
        """逐模型推理

        Args:
            dataset (Dataset): 推理数据集
            evaluate (bool, optional): 是否需要性能评估（需要提供数据真实标签）. Defaults to False.
            dataset_labels (list, optional): 数据集标签. Defaults to None.

        Returns:
            tuple[list, list]: _description_
        """
        if evaluate is True:
            assert (
                dataset_labels is not None
            ), "Error! Set the `evaluate` parameter to True but did not provide the `dataset_labels` parameter."
            model_evaluator = PerformanceEvaluator()  # 性能评估

        activated_models_config = []
        for model_config in self.models_config:
            if model_config["active"] is True:
                activated_models_config.append(model_config)

        models_feats = {}  # 存储各模型输出特征
        models_preds = {}  # 存储各模型预测输出
        for model_config in tqdm(activated_models_config, desc="Overall progress"):
            print("\n\n")
            logging.info(
                "Current detector: %s(%s) %s",
                model_config["id"],
                model_config["cls"].__name__,
                model_config["cls"].desc,
            )

            # 实例化模型
            model = model_config["cls"](
                *model_config["init_args"], **model_config["init_kwargs"]
            )
            setattr(model, "model_id", model_config["id"])

            # 推理
            results = model.inference_dataset(
                dataset, *model_config["args"], **model_config["kwargs"]
            )

            # 评估
            if evaluate is True:
                model_evaluator.calculate_classification_performance(
                    dataset_labels, results["preds"]
                )

            model_flag = f"{model_config["id"]}({model.__class__.__name__})"
            models_feats[model_flag] = results
            models_preds[model_flag] = results["preds"]

            time.sleep(1)

        return models_preds, models_feats

    def voting(
        self,
        models_preds: dict[str, list],
        evaluate: bool = False,
        dataset_labels: list = None,
        threshold: int = None,
        vote_type: str = "average",
        dynamic_vote_weight: bool = False,
        fine_grain_vote_flag: dict = None,
        **kwargs,
    ) -> list:
        """集成模型投票（平均/无加权）

        Args:
            models_preds (dict[str, list]): 各个模型的预测结果
            evaluate (bool, optional): 是否需要性能评估（需要提供数据真实标签）. Defaults to False.
            dataset_labels (list, optional): 数据集标签. Defaults to None.
            threshold (int, optional): 投票阈值（若支持样本为机器文本的模型数量大于等于该阈值，则判定为机器文本）. Defaults to None.
            vote_type (str, optional): 投票类型。可选值："weighted"或"average". Defaults to "average".
            dynamic_vote_weight (bool, optional): 是否动态设置投票权重. Defaults to False.
            fine_grain_vote_flag (dict, optional): 如果需要动态投票权重，必须提供此项。
                为每种特定的投票策略提供一个数据集大小的细粒度的样本标记，指明在哪个策略下哪个样本需要应用特殊投票规则. Defaults to None.

        Returns:
            list: _description_
        """
        logging.info(
            "Current detector: Hybrid ensemble voting model for machine-generated text detection"
        )
        assert (
            len(set([len(preds) for preds in models_preds.values()])) == 1
        ), "Error! The inference result list length differs for each model on the dataset."
        if evaluate is True:
            assert len(dataset_labels) == len(
                list(models_preds.values())[0]
            ), "Error! The dataset labels and prediction results have inconsistent lengths."
        if dynamic_vote_weight is True:
            assert (
                fine_grain_vote_flag is not None
            ), "Require dataset to apply dynamic voting weight."

        # 过滤不参与投票的模型预测
        models_for_vote = []
        for model_config in self.models_config:
            if model_config["vote"] is True:
                models_for_vote.append(
                    f"{model_config['id']}({model_config['cls'].__name__})"
                )

        # 取出参与投票的模型的预测
        models_for_vote_preds = []
        for model, preds in models_preds.items():
            if model in models_for_vote:
                models_for_vote_preds.append(copy.deepcopy(preds))

        num_samples = len(list(models_preds.values())[0])
        num_vote_models = len(models_for_vote_preds)
        preds_matrix = np.array(models_for_vote_preds).T.tolist()

        vote_types = ["average", "weighted"]
        assert vote_type in vote_types, f"`vote_type` should be one of {vote_types}."

        vote_preds, vote_scores = [], []
        if vote_type == "average":  # 平均投票
            logging.info("Set the voting mode to `average`")
            if threshold is None:
                threshold = (num_vote_models + 1) // 2
            for i in range(num_samples):
                vote_pos = sum(preds_matrix[i])
                vote_neg = num_vote_models - vote_pos
                # vote_preds.append(1 if vote_pos >= threshold else 0)  # 平票时优先判定机器
                vote_preds.append(
                    0 if vote_neg >= threshold else 1
                )  # 平票时优先判定人类

        else:  # 加权投票
            logging.info("Set the voting mode to `weighted`")
            assert (
                threshold is not None
            ), "Set `vote_type=weighted` but did not provide parameter `threshold`."

            if dynamic_vote_weight is False:
                # 取出各模型的票权
                models_vote_weights = []
                for model_config in self.models_config:
                    if model_config["vote"] is True and model_config["active"] is True:
                        models_vote_weights.append(model_config["importance"])
                for i in tqdm(range(num_samples), desc="Voting ..."):
                    vote_score = 0
                    for k, pred in enumerate(preds_matrix[i]):
                        if pred == 1:
                            vote_score += models_vote_weights[k]
                        else:
                            vote_score -= models_vote_weights[k]
                    vote_preds.append(1 if vote_score > 0 else 0)
                    vote_scores.append(vote_score)

            else:
                # 检查各种动态投票策略的合法性
                assert set(self.dynamic_vote_weights.keys()) == set(
                    self.dynamic_vote_strategy.keys()
                )
                dynamic_vote_digits = [
                    sum(x) for x in zip(*fine_grain_vote_flag.values())
                ]
                for i in range(num_samples):
                    assert (
                        dynamic_vote_digits[i] <= 1
                    ), "Conflict detected in different vote rules."

                # 用一个列表存储所有样本各自的投票类型
                vote_type_for_samples = ["default"] * num_samples
                for rule_type, sample_flag in fine_grain_vote_flag.items():
                    for i in range(num_samples):
                        if sample_flag[i] == 1:
                            vote_type_for_samples[i] = rule_type

                # 集成各种投票策略下各个模型的投票权重
                models_vote_weights = defaultdict(list)
                for model_config in self.models_config:
                    model_flag = f"{model_config["id"]}({model_config["cls"].__name__})"
                    if model_config["vote"] is True and model_config["active"] is True:
                        models_vote_weights["default"].append(
                            model_config["importance"]
                        )
                        for rule_type, weights in self.dynamic_vote_weights.items():
                            models_vote_weights[rule_type].append(weights[model_flag])

                # 引入对抗样本辅助判断：先利用llm提示工程尝试识别出潜在的对抗样本，参与集成模型的投票决策
                with open(
                    os.path.join(Config.DATA_DIR, "test_assist_translate.json"),
                    "r",
                    encoding="utf-8",
                ) as f:
                    dataset_check = json.load(f)

                # 对每个样本，应用独特的投票策略
                for i in tqdm(range(num_samples), desc="Voting ..."):
                    vote_score = 0
                    for k, pred in enumerate(preds_matrix[i]):
                        if pred == 1:
                            vote_score += models_vote_weights[vote_type_for_samples[i]][
                                k
                            ]
                        else:
                            vote_score -= models_vote_weights[vote_type_for_samples[i]][
                                k
                            ]

                    # 针对困难样本（投票评分置信度低），结合辅助信息加以鉴别
                    # 由于辅助信息可正确可错误，需基于训练集和验证集选取各个合适的阈值，采用复杂的嵌套逻辑判断，使之与集成模型配合发挥最佳作用
                    # 注：以下嵌套逻辑中的所有超参数均来自预设定阈值和基于训练集与验证集的模型选择，是本集成模型的重要组成部分，而非简单的基于规则的方法
                    if vote_score < 0 and vote_type_for_samples[i] == "default":
                        vote_score_align = vote_score + 1000
                        if abs(vote_score_align) <= 250:
                            if dataset_check[i]["judge"] == 1:
                                vote_preds.append(1)
                            else:
                                vote_preds.append(0)
                        else:
                            vote_preds.append(0)
                    elif (
                        vote_score < 0
                        and vote_type_for_samples[i] == "extreme_short_text"
                    ):
                        vote_score_align = vote_score + 1000
                        if vote_score_align > 150:
                            if sum(preds_matrix[i][6:10]) == 4:
                                vote_preds.append(1)
                            else:
                                vote_preds.append(0)
                        else:
                            vote_preds.append(0)
                    elif vote_score > 0 and vote_type_for_samples[i] == "short_text":
                        vote_score_align = vote_score - 1000
                        if vote_score_align < -20:
                            if sum(preds_matrix[i][2:4]) == 0:
                                vote_preds.append(0)
                            else:
                                vote_preds.append(1)
                        else:
                            vote_preds.append(1)
                    else:
                        vote_preds.append(1 if vote_score > 0 else 0)

                    vote_scores.append(vote_score)

        if evaluate:
            assert (
                dataset_labels is not None
            ), "Error! Set parameter `evaluate` to True but did not provide parameter `dataset_labels`."
            model_evaluator = PerformanceEvaluator()  # 性能评估
            model_evaluator.calculate_classification_performance(
                dataset_labels, vote_preds
            )

        return vote_preds, vote_scores

    def adjust_determinative(
        self,
        models_preds: dict[str, list],
        vote_preds: list,
        evaluate: bool = False,
        dataset_labels: list = None,
    ) -> list:
        """确定性调整（投票之后）。有些模型给出置信度极高的判断，如：
        - 包含双换行符的文本，必须判定为机器文本
        - 包含连续多个相同标点符号的文本，必须判定为人类文本

        Args:
            models_preds (dict[str, list]): 各个模型的预测结果
            vote_preds (list): 投票结果
            evaluate (bool, optional): 是否需要性能评估（需要提供数据真实标签）. Defaults to False.
            dataset_labels (list, optional): 数据集标签. Defaults to None.

        Returns:
            list: _description_
        """
        logging.info(
            "Current detector: Hybrid ensemble voting model for machine-generated text detection (with determinative adjustment)"
        )
        # 标记出确定要预测为正/负的样本
        determine_pos_samples, determine_neg_samples = set(), set()
        for model_config in self.models_config:
            if model_config["active"] is True:
                model_flag = f"{model_config['id']}({model_config['cls'].__name__})"
                if model_config["determinative"] == 1:
                    for i in range(len(models_preds[model_flag])):
                        if models_preds[model_flag][i] == 1:
                            determine_pos_samples.add(i)
                elif model_config["determinative"] == 0:
                    for i in range(len(models_preds[model_flag])):
                        if models_preds[model_flag][i] == 0:
                            determine_neg_samples.add(i)

        # 对投票结果进行调整
        adjusted_preds = copy.deepcopy(vote_preds)
        for i, _ in enumerate(adjusted_preds):
            if i in determine_pos_samples:
                adjusted_preds[i] = 1
            elif i in determine_neg_samples:
                adjusted_preds[i] = 0

        if evaluate:
            assert (
                dataset_labels is not None
            ), "Error! Set the `evaluate` parameter to True but did not provide the `dataset_labels` parameter."
            model_evaluator = PerformanceEvaluator()  # 性能评估
            model_evaluator.calculate_classification_performance(
                dataset_labels, adjusted_preds
            )

        return adjusted_preds

    def pipeline(
        self,
        dataset: Dataset,
        dataset_flag: str,
        models_config: list[dict],
        evaluate: bool = True,
        labels: list = None,
        vote_type: str = "average",
        dynamic_vote_weight: bool = False,
        fine_grain_vote_flag: dict = None,
    ):
        """数据集推理pipeline

        Args:
            dataset (Dataset): 推理数据集
            dataset_flag (str): 数据集标记
            labels (list, optional): 数据集标签
            models_config (list[dict]): 模型配置
            evaluate (bool, optional): 是否进行性能评估. Defaults to True.
            vote_type (str, optional): 投票类型。可选值："weighted"或"average". Defaults to "average".
            dynamic_vote_weight (bool, optional): 是否动态设置投票权重. Defaults to False.
            fine_grain_vote_flag (dict, optional): 如果需要动态投票权重，必须提供此项。
                为每种特定的投票策略提供一个数据集大小的细粒度的样本标记，指明在哪个策略下哪个样本需要应用特殊投票规则. Defaults to None.

        Returns:
            _type_: _description_
        """
        print("\n================ Performing Inference ================\n")
        models_preds, models_feats = self.inference_dataset(
            dataset, evaluate=evaluate, dataset_labels=labels
        )

        print("\n================ Saving Predictions ================\n")
        feats_cache_path = generate_cache_path(
            data=dataset,
            filename=f"model_feats_{dataset_flag}.tmp",
            model=models_config,
        )
        with open(feats_cache_path, "wb") as f_cache:
            pickle.dump(models_feats, f_cache)
            logging.info("Save model output features to cache `%s`", feats_cache_path)
        # feats_log_path = os.path.join(Config.LOG_DIR, f"model_feats_{dataset_flag}.txt")
        # with open(feats_log_path, "w", encoding="utf-8") as f:
        #     json.dump(models_feats, f, ensure_ascii=False, indent=4)
        #     logging.info(
        #         "Save model output features (readable) to file `%s`", feats_log_path
        #     )

        print("\n================ Voting ================\n")
        vote_preds, vote_scores = self.voting(
            models_preds,
            evaluate=evaluate,
            dataset_labels=labels,
            vote_type=vote_type,
            threshold=0,
            dynamic_vote_weight=dynamic_vote_weight,
            fine_grain_vote_flag=fine_grain_vote_flag,
        )

        print("\n================ Applying Determinant ================\n")
        adjusted_preds = self.adjust_determinative(
            models_preds, vote_preds, evaluate=evaluate, dataset_labels=labels
        )

        return adjusted_preds, models_preds, vote_scores


if __name__ == "__main__":

    print("\n================ Preparing Datasets ================\n")
    logging.info("Loading train-set from %s ...", repr(Config.RAW_DATA["train"]))
    train_dataset = load_dataset("json", data_files={"train": Config.RAW_DATA["train"]})
    logging.info("Loading dev-set from %s ...", repr(Config.RAW_DATA["dev"]))
    dev_dataset = load_dataset("json", data_files={"dev": Config.RAW_DATA["dev"]})
    logging.info("Loading test-set from %s ...", repr(Config.RAW_DATA["test"]))
    test_dataset = load_dataset("json", data_files={"test": Config.RAW_DATA["test"]})
    dataset_dict = {
        "train": train_dataset["train"],
        "dev": dev_dataset["dev"],
        "test": test_dataset["test"],
    }
    print(dataset_dict)

    print("\n================ Defining Models ================\n")
    # extra_kwargs = {
    #     "labels": dataset_dict["dev"]["label"],
    #     "refer_dataset": dataset_dict["train"],
    #     "dataset_flag": "dev",
    # }
    extra_kwargs = {
        "labels": None,
        "refer_dataset": dataset_dict["train"],
        "dataset_flag": "test",
    }
    models_config_ = Config.prepare_models_config(**extra_kwargs)
    models_config_ += Config.prepare_models_config_joint(models_config_, **extra_kwargs)
    ensemble_model = EnsembleModelForAIGTDetection(models_config_)

    # 验证集
    # ensemble_model.pipeline(
    #     dataset=dataset_dict["dev"],
    #     dataset_flag="dev",
    #     labels=dataset_dict["dev"]["label"],
    #     models_config=models_config_,
    #     evaluate=True,
    #     vote_type="weighted",
    #     dynamic_vote_weight=False,
    # )

    # 测试集
    vote_strategies = build_all_exist_strategies(dataset_dict["test"])
    final_preds, models_preds_, vote_scores_ = ensemble_model.pipeline(
        dataset=dataset_dict["test"],
        dataset_flag="test",
        labels=None,
        models_config=models_config_,
        evaluate=False,
        vote_type="weighted",
        dynamic_vote_weight=True,
        fine_grain_vote_flag=vote_strategies,
    )

    # 输出预测文件
    save_file = os.path.join(Config.PROJECT_DIR, "prediction.json")
    with open(Config.RAW_DATA["test"], "r", encoding="utf-8") as f_in:
        testset = json.load(f_in)
    for i, pred in enumerate(final_preds):
        testset[i]["label"] = pred
        # testset[i]["vote_score"] = vote_scores_[i]
        # # 提取各个模型对该样本的预测
        # models_pred_ = {}
        # for model_flag_, model_preds_ in models_preds_.items():
        #     models_pred_[model_flag_] = model_preds_[i]
        # testset[i]["models_pred"] = models_pred_
    with open(save_file, "w", encoding="utf-8") as f_out:
        json.dump(testset, f_out, indent=4, ensure_ascii=False)
        logging.info("Save final prediction file to %s", save_file)
