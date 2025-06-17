# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os
import random
import json
from detector.config import Config


def divide_text(text: str):
    sep = ["。", "！", "？"]
    sep_ids = []
    for idx, ch in enumerate(text):
        if ch in sep:
            sep_ids.append(idx)
    return sep_ids


if __name__ == "__main__":

    with open(Config.RAW_DATA["dev"], "r", encoding="utf-8") as f:
        train_data_raw = json.load(f)

    len_range = {
        "extreme_short_text": {"min": 55, "max": 75},
        "short_text": {"min": 110, "max": 150},
    }

    extreme_short_text = {"1": set(), "0": set()}
    short_text = {"1": set(), "0": set()}

    for idx, example in enumerate(train_data_raw):
        sep_ids = divide_text(example["text"])
        for i, _ in enumerate(sep_ids):
            for j in range(i + 1, len(sep_ids)):
                if (
                    len_range["extreme_short_text"]["min"]
                    <= sep_ids[j] - sep_ids[i]
                    <= len_range["extreme_short_text"]["max"]
                    and sep_ids[j] - sep_ids[j - 1] > 1
                ):
                    extreme_short_text[str(example["label"])].add(
                        example["text"][sep_ids[i] + 1 : sep_ids[j] + 1]
                    )
                if (
                    len_range["short_text"]["min"]
                    <= sep_ids[j] - sep_ids[i]
                    <= len_range["short_text"]["max"]
                    and sep_ids[j] - sep_ids[j - 1] > 1
                ):
                    short_text[str(example["label"])].add(
                        example["text"][sep_ids[i] + 1 : sep_ids[j] + 1]
                    )

    print(
        f"size of short text collection: label=0: {len(short_text["0"])}, label=1: {len(short_text["1"])}"
    )
    print(
        f"size of extreme short text collection: label=0: {len(extreme_short_text["0"])}, label=1: {len(extreme_short_text["1"])}"
    )

    sample_size_for_each_label = 2000
    sample_short_text = random.sample(
        list(short_text["0"]), sample_size_for_each_label
    ) + random.sample(list(short_text["1"]), sample_size_for_each_label)
    sample_extreme_short_text = random.sample(
        list(extreme_short_text["0"]), sample_size_for_each_label
    ) + random.sample(list(extreme_short_text["1"]), sample_size_for_each_label)

    short_text_dataset = []
    for i in range(2 * sample_size_for_each_label):
        if i < sample_size_for_each_label:
            short_text_dataset.append(
                {
                    "text": sample_short_text[i],
                    "label": 0,
                }
            )
        else:
            short_text_dataset.append(
                {
                    "text": sample_short_text[i],
                    "label": 1,
                }
            )
    print(len(short_text_dataset))
    with open(
        os.path.join(Config.DATA_DIR, "data_augment/dev_short.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(short_text_dataset, f, ensure_ascii=False, indent=4)

    extreme_short_text_dataset = []
    for i in range(2 * sample_size_for_each_label):
        if i < sample_size_for_each_label:
            extreme_short_text_dataset.append(
                {
                    "text": sample_extreme_short_text[i],
                    "label": 0,
                }
            )
        else:
            extreme_short_text_dataset.append(
                {
                    "text": sample_extreme_short_text[i],
                    "label": 1,
                }
            )
    print(len(extreme_short_text_dataset))
    with open(
        os.path.join(Config.DATA_DIR, "data_augment/dev_extreme_short.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(extreme_short_text_dataset, f, ensure_ascii=False, indent=4)
