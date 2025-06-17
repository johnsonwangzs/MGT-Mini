# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os
import json
import random
from detector.config import Config


with open(
    os.path.join(Config.DATA_DIR, "data_augment/train_short.json"),
    "r",
    encoding="utf-8",
) as f_raw:
    raw_data = json.load(f_raw)

# raw_data = random.sample(raw_data, k=10000)

sft_data = []

for i, example in enumerate(raw_data):
    system = "你是一个善于区分机器生成文本和人类撰写文本的专家。"
    instruction = "请判断以下文本是由人类撰写还是由机器生成："
    input_ = example["text"].replace("\n\n", "")  # 删去所有双换行符
    output_ = (
        "这段文本是机器生成的。" if example["label"] == 1 else "这段文本是人类撰写的。"
    )
    sft_example = {
        "system": system,
        "instruction": instruction,
        "input": input_,
        "output": output_,
    }
    sft_data.append(sft_example)

print(len(sft_data))

with open(
    os.path.join(Config.DATA_DIR, "data_sft/nlpcc25_task1_train_short.json"),
    "w",
    encoding="utf-8",
) as f_sft:
    json.dump(sft_data, f_sft, ensure_ascii=False, indent=4)

print("Done.")
