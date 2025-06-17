# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os
import json
import re
import numpy as np
from scipy import stats
import statistics
from collections import defaultdict
from transformers import AutoTokenizer
from rich.progress import track
from detector.config import Config


def count_common_phrase(dataset):
    rule_patterns = [
        r"总的来说，",
        r"总体来说，",
        r"总体而言，",
        r"总而言之，",
        r"总之，",
        r"总是",
    ]
    for rule_pattern in rule_patterns:
        cnt_1, cnt_0 = 0, 0
        for example in dataset:
            if rule_pattern in example["text"]:
                if example["label"] == 1:
                    cnt_1 += 1
                else:
                    cnt_0 += 1
        print(f"{rule_pattern=}   human: {cnt_0}   machine: {cnt_1}")


def count_consecutive_punctuation(dataset):
    rule_patterns = [
        r"！！！",
        r"。。。",
    ]
    for rule_pattern in rule_patterns:
        cnt_1, cnt_0 = 0, 0
        for example in dataset:
            if rule_pattern in example["text"]:
                if example["label"] == 1:
                    cnt_1 += 1
                else:
                    cnt_0 += 1
        print(f"{rule_pattern=}   human: {cnt_0}   machine: {cnt_1}")


def print_raw_data_base_info(dataset):
    """打印数据集基本信息"""
    cnt_source = defaultdict(int)
    cnt_model = defaultdict(int)
    cnt_label = defaultdict(int)
    cnt_human = defaultdict(int)
    human_examples = []
    ai_examples = []
    for example in track(dataset, description="正在统计原始数据集基本信息..."):
        cnt_source[example["source"]] += 1
        cnt_model[example["model"]] += 1
        cnt_label[example["label"]] += 1
        if example["model"] == "human":
            cnt_human["human_" + example["source"]] += 1
            human_examples.append(example)
        else:
            ai_examples.append(example)

    print(f"数据总量：{len(dataset)}")
    print(f"各来源计数：{cnt_source}")
    print(f"各模型样本计数：{cnt_model}")
    print(f"各标签计数：{cnt_label}")
    print(f"人类样本来源计数:{cnt_human}")

    return human_examples, ai_examples


def base_feature_extract(texts, shard_id):
    """统计样本的基本信息，获取基本特征"""
    model_name = "glm-4-9b-chat"
    tokenizer = AutoTokenizer.from_pretrained(
        os.path.join(Config.MODEL_DIR, model_name), trust_remote_code=True
    )

    # (总token数，标点符号数，单个\n出现次数，两个\n出现次数，三个\n出现次数)
    statistics = []
    punctuation_pattern = r"[,，.。!！?？:：、]"
    single_newline_pattern = r"[^\n][\n][^\n]"
    double_newline_pattern = r"[^\n]\n{2}[^\n]"
    multi_newline_pattern = r"\n{3}"
    double_sharp_pattern = r"#{2}"
    for example in track(texts, description=f"正在统计{shard_id}文本基本特征..."):
        result = tokenizer(example, add_special_tokens=False, return_length=True)

        token_cnt_total = result["length"]
        token_cnt_punctuation = len(re.findall(punctuation_pattern, example))

        single_newline_cnt = len(re.findall(single_newline_pattern, example))
        double_newline_cnt = len(re.findall(double_newline_pattern, example))
        multi_newline_cnt = len(re.findall(multi_newline_pattern, example))
        double_sharp_cnt = len(re.findall(double_sharp_pattern, example))

        statistics.append(
            (
                token_cnt_total,
                token_cnt_punctuation,
                single_newline_cnt,
                double_newline_cnt,
                multi_newline_cnt,
                double_sharp_cnt,
            )
        )

    print(f"已统计：{len(statistics)}条")
    return statistics


def base_feature_analysis(dataset, shard_id=None):
    """分析样本的基本特征"""

    texts = [example["text"] for example in dataset]

    base_statistics = base_feature_extract(texts, shard_id)

    punctuation_ratio = list(map(lambda x: x[1] / x[0], base_statistics))
    punctuation_ratio_avg = sum(punctuation_ratio) / len(punctuation_ratio)
    print(f"{shard_id}文本，标点符号占总token比例：{punctuation_ratio_avg}")

    single_newline_cnt, double_newline_cnt, multi_newline_cnt, double_sharp_cnt = (
        0,
        0,
        0,
        0,
    )
    for item in base_statistics:
        is_single_newline, is_double_newline, is_multi_newline, is_double_sharp = (
            item[2],
            item[3],
            item[4],
            item[5],
        )
        single_newline_cnt += 1 if is_single_newline > 0 else 0
        double_newline_cnt += 1 if is_double_newline > 0 else 0
        multi_newline_cnt += 1 if is_multi_newline > 0 else 0
        double_sharp_cnt += 1 if is_double_sharp > 0 else 0
    single_newline_prob = single_newline_cnt / len(texts)
    double_newline_prob = double_newline_cnt / len(texts)
    multi_newline_prob = multi_newline_cnt / len(texts)
    double_sharp_prob = double_sharp_cnt / len(texts)
    print(f"{shard_id}文本，出现单个\\n概率：{single_newline_prob}")
    print(f"{shard_id}文本，出现\\n\\n概率：{double_newline_prob}")
    print(f"{shard_id}文本，出现多个\\n概率：{multi_newline_prob}")
    print(f"{shard_id}文本，出现##概率：{double_sharp_prob}")


def count_subsentence(data: str | list):
    """统计文本中的子句数量"""
    if isinstance(data, str):
        matches = re.findall(r"[。！？]", data)
    if isinstance(data, list):
        matches = []
        for example in data:
            matches.append(len(re.findall(r"[。！？]", example)))
    return matches


def show_stat(data: list):
    """统计信息

    Args:
        data (list): 在一个数据集上计算的曲率分数
    """
    # 基本统计量
    mean = statistics.mean(data)
    median = statistics.median(data)
    stdev = statistics.stdev(data)
    variance = statistics.variance(data)
    minimum = min(data)
    maximum = max(data)
    data_range = maximum - minimum

    # 使用numpy补充的一些统计量
    percentiles = np.percentile(data, [25, 50, 75])  # 四分位数
    skewness = stats.skew(data)  # 偏度
    kurtosis = stats.kurtosis(data)  # 峰度

    # 输出结果
    print(f"均值: {mean:.2f}")
    print(f"中位数: {median:.2f}")
    print(f"标准差: {stdev:.2f}")
    print(f"方差: {variance:.2f}")
    print(f"最小值: {minimum}")
    print(f"最大值: {maximum}")
    print(f"极差: {data_range:.2f}")
    print(
        f"四分位数: Q1={percentiles[0]:.2f}, Q2={percentiles[1]:.2f}, Q3={percentiles[2]:.2f}"
    )
    print(f"偏度: {skewness:.2f}")
    print(f"峰度: {kurtosis:.2f}")


if __name__ == "__main__":
    print(Config.RAW_DATA)
    with open(Config.RAW_DATA["train"], "r", encoding="utf-8") as f_train:
        train_raw = json.load(f_train)

    human_data, ai_data = print_raw_data_base_info(train_raw)

    # base_feature_analysis(human_data, shard_id="human")
    # base_feature_analysis(ai_data, shard_id="ai")

    count_common_phrase(train_raw)

    count_consecutive_punctuation(train_raw)
