# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import logging
import json
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    roc_curve,
    auc,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class PerformanceEvaluator:


    def calculate_classification_performance(
        self,
        labels: list,
        preds: list,
        scores: list = None,
        max_fpr: float = 0.01,
        print_result: bool = True,
    ):

        logging.info("Evaluating classification performance ...")
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds)
        recall = recall_score(labels, preds)
        f1 = f1_score(labels, preds)

        if scores is not None:
            tpr_at_fpr, _ = self.tpr_at_fpr(labels, scores, max_fpr=max_fpr)

        if print_result:
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-score: {f1:.4f}")

            if scores is not None:
                print(f"TPR@FPR={max_fpr}: {tpr_at_fpr:.4f}")

            print(classification_report(labels, preds, digits=4))

        return precision, recall, f1

    @classmethod
    def find_best_threshold_f1(cls, y_true: list, y_scores: list):

        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # 计算 F1 分数，注意 precision 和 recall 比 thresholds 多一个点（threshold=-∞时的点）
        f1 = 2 * (precision[1:] * recall[1:]) / (precision[1:] + recall[1:] + 1e-8)
        best_idx = np.argmax(f1)
        return thresholds[best_idx], f1[best_idx]

    def tpr_at_fpr(self, y_true: list, y_scores: list, max_fpr: float = 0.01):

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)

        # 找出所有 FPR 小于等于 max_fpr 的索引
        valid_idxs = np.where(fpr <= max_fpr)[0]

        if len(valid_idxs) == 0:
            return 0.0, None  # 没有满足条件的阈值点

        best_idx = np.argmax(tpr[valid_idxs])
        best_tpr = tpr[valid_idxs][best_idx]
        best_threshold = thresholds[valid_idxs][best_idx]

        return best_tpr, best_threshold

    def draw_roc(self, y_true: list, y_scores: list, save_path: str):

        # 计算 ROC 曲线
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # 绘图
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})", linewidth=2)
        plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")  # 对角线
        plt.axvline(
            x=0.01, color="red", linestyle="--", label="FPR = 0.01"
        )  # TPR@FPR线

        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.grid(True)

        # 保存图像
        plt.savefig(save_path, dpi=300)
        # plt.show()

    def evaluate_from_predict_file(self, file_path: str):
        with open(file_path, "r", encoding="utf-8") as f:
            results = f.readlines()

        preds, labels = [], []

        for line in results:
            json_obj = json.loads(line)
            pred = json_obj["predict"]
            label = json_obj["label"]

            preds.append(1 if "机器" in pred else 0)
            labels.append(1 if "机器" in label else 0)

        self.calculate_classification_performance(labels, preds)
