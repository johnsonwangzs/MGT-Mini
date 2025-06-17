# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import logging
import copy
import pickle
from datasets import load_dataset, Dataset
from detector.config import Config
from detector.utils import generate_cache_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def build_extreme_short_text_strategy(dataset: Dataset) -> list:

    logging.info("Generating `extreme_short_text` strategy ...")
    strategy = []
    for example in dataset["text"]:
        strategy.append(1 if len(example) <= 75 else 0)
    return strategy


def build_short_text_strategy(dataset: Dataset) -> list:

    logging.info("Generating `short_text` strategy ...")
    strategy = []
    for example in dataset["text"]:
        strategy.append(1 if 75 < len(example) <= 180 else 0)
    return strategy


def build_medium_text_strategy(dataset: Dataset) -> list:

    logging.info("Generating `medium_text` strategy ...")
    strategy = []
    for example in dataset["text"]:
        strategy.append(1 if 180 < len(example) < 300 else 0)
    return strategy


def build_all_exist_strategies(dataset: Dataset) -> dict[str, list]:

    # 投票策略
    vote_strategies = {}

    vote_strategies["extreme_short_text"] = copy.deepcopy(
        build_extreme_short_text_strategy(dataset)
    )

    vote_strategies["short_text"] = copy.deepcopy(build_short_text_strategy(dataset))
    vote_strategies["medium_text"] = copy.deepcopy(build_medium_text_strategy(dataset))

    vote_strategies_path = generate_cache_path(
        data=vote_strategies,
        filename="vote_strategies.tmp",
        dataset=dataset,
    )
    with open(vote_strategies_path, "wb") as f_cache:
        pickle.dump(vote_strategies, f_cache)
        logging.info("Save vote strategies to %s", vote_strategies_path)

    return vote_strategies


if __name__ == "__main__":

    logging.info("Loading test-set from %s ...", repr(Config.RAW_DATA["test"]))
    test_dataset = load_dataset("json", data_files={"test": Config.RAW_DATA["test"]})
    dataset_dict = {"test": test_dataset["test"]}

    build_all_exist_strategies(dataset_dict["test"])
