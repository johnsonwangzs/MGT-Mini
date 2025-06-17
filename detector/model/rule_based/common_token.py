# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os.path
import logging
import pickle
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from datasets import load_dataset, Dataset
from detector.config import Config
from detector.utils import generate_cache_path
from detector.model.rule_based.base_rule_model import BaseRuleModel
from detector.eval.evaluator import PerformanceEvaluator

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


class CommonTokenModel(BaseRuleModel):
    """对于待测文本，比较其 tokens 在人类偏好词频表和机器偏好词频表中的出现数量"""

    desc = repr("The machine regularly generates some representative tokens")

    def __init__(
        self,
        tokenizer_path: str,
        build_token_dict: bool = True,
        refer_dataset: Dataset = None,
        human_token_dict: Counter = None,
        machine_token_dict: Counter = None,
    ):
        super().__init__()
        logging.info("Using tokenizer from `%s`", tokenizer_path)
        self.tokenizer_path = tokenizer_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, trust_remote_code=True
        )

        if build_token_dict is True:
            self.machine_token_dict = self.build_tokens_dict(
                refer_dataset, target_label=1
            )
            self.human_token_dict = self.build_tokens_dict(
                refer_dataset, target_label=0
            )
        else:
            assert (
                human_token_dict is not None and machine_token_dict is not None
            ), "Parameter `build_token_dict` is set to `False`, but `human_token_dict` and `machine_token_dict` are not provided."
            self.machine_token_dict = machine_token_dict
            self.human_token_dict = human_token_dict

    def inference_dataset(
        self,
        dataset: Dataset,
        top_k: int = 1000,
        **kwargs,
    ):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        logging.info("Set top-k to %s", top_k)
        cache_path = generate_cache_path(
            data=dataset,
            filename=f"{self.__class__.__name__}_preds.tmp",
            model_name=self.tokenizer_path,
            top_k=top_k,
        )
        if not os.path.exists(cache_path):
            machine_token_topk = [
                token[0] for token in self.machine_token_dict.most_common(top_k)
            ]
            human_token_topk = [
                token[0] for token in self.human_token_dict.most_common(top_k)
            ]

            # 统计文本命中人类top-k和机器top-k的命中率
            preds, feats = [], []
            for example in tqdm(dataset["text"], desc="Inferring ..."):
                hit_machine_token, hit_human_token = 0, 0
                token_seq = self._tokenize(example)
                for token in token_seq:
                    if token in machine_token_topk:
                        hit_machine_token += 1
                    if token in human_token_topk:
                        hit_human_token += 1
                preds.append(0 if hit_human_token > hit_machine_token else 1)
                feats.append([hit_human_token, hit_machine_token])

            cache_data = {"preds": preds, "hit_token_cnt": feats}
            with open(cache_path, "wb") as f_cache:
                pickle.dump(cache_data, f_cache)
                logging.info("Save model prediction results to cache `%s`", cache_path)
        else:
            logging.info("Reading prediction results from cache `%s`", cache_path)
            with open(cache_path, "rb") as f_cache:
                cache_data = pickle.load(f_cache)

        return cache_data

    def inference_example(self, text, **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        pass

    def __call__(self, text):
        return self.inference_example(text)

    def _tokenize(self, text: str):
        token_ids = self.tokenizer.encode(text)
        decoded_tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        return decoded_tokens

    def build_tokens_dict(self, dataset: Dataset, target_label: int) -> Counter:
        """创建tokens频率统计字典，记录数据集中每个token出现的次数

        Args:
            dataset (Dataset): 源数据集
            target_label (int): 统计的标签。统计人类文本（0）或机器文本（1）

        Returns:
            Counter: tokens频率统计字典
        """

        logging.info(
            "Building token dict of texts with label=%s in dataset ...", target_label
        )

        cache_path = generate_cache_path(
            data=dataset,
            filename=f"{self.__class__.__name__}_{target_label}.tmp",
            model_name=self.tokenizer_path,
        )
        if not os.path.exists(cache_path):
            tokens_dict = Counter()
            for example in tqdm(dataset, desc="tokenizing..."):
                if example["label"] == target_label:
                    text = example["text"]
                    token_seq = self._tokenize(text)
                    tokens_dict.update(token_seq)
            with open(cache_path, "wb") as f_cache:
                pickle.dump(tokens_dict, f_cache)
                logging.info("Save token dict to cache `%s`", cache_path)
        else:
            logging.info("Reading token dict from cache `%s`", cache_path)
            with open(cache_path, "rb") as f_cache:
                tokens_dict = pickle.load(f_cache)

        return tokens_dict

    def draw_plot(self, dataset, labels, start_topk, end_topk, step):
        evaluator = PerformanceEvaluator()  # 性能评估

        tokenized_dataset = []
        for example in tqdm(dataset["text"], desc="Tokenizing ..."):
            tokenized_dataset.append(self._tokenize(example))

        top_ks, f1s, precisions, recalls = [], [], [], []
        for top_k in tqdm(
            range(start_topk, end_topk, step), desc="Traversing top-k ..."
        ):

            machine_token_topk = [
                token[0] for token in self.machine_token_dict.most_common(top_k)
            ]
            human_token_topk = [
                token[0] for token in self.human_token_dict.most_common(top_k)
            ]

            # 统计文本命中人类top-k和机器top-k的命中率
            preds = []
            for token_seq in tokenized_dataset:
                hit_machine_token, hit_human_token = 0, 0
                for token in token_seq:
                    if token in machine_token_topk:
                        hit_machine_token += 1
                    if token in human_token_topk:
                        hit_human_token += 1
                preds.append(0 if hit_human_token > hit_machine_token else 1)

            top_ks.append(top_k)
            precision, recall, f1 = evaluator.calculate_classification_performance(
                labels, preds, print_result=False
            )
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)

        plt.figure(figsize=(16, 10))
        plt.plot(top_ks, f1s, marker="o", linestyle="-", label="f1")
        plt.plot(top_ks, precisions, marker="o", linestyle="-", label="precision")
        plt.plot(top_ks, recalls, marker="o", linestyle="-", label="recall")
        plt.xlabel("top-k")
        plt.ylabel("metric")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.__class__.__name__}.png", dpi=300, bbox_inches="tight")
        plt.close()
        # plt.show()


if __name__ == "__main__":

    logging.info("Loading dataset ...")
    trainset_path = os.path.join(Config.RAW_DATA["train"])
    devset_path = os.path.join(Config.RAW_DATA["dev"])
    dataset_dict = load_dataset(
        "json", data_files={"train": trainset_path, "dev": devset_path}
    )
    print(dataset_dict)

    tokenizer_path = os.path.join(Config.CKPT_DIR, "glm-4-9b-chat")

    common_token_model = CommonTokenModel(
        tokenizer_path, build_token_dict=True, refer_dataset=dataset_dict["train"]
    )
    # machine_tokens_dict = model.build_tokens_dict(dataset_dict["train"], target_label=1)
    # human_tokens_dict = model.build_tokens_dict(dataset_dict["train"], target_label=0)

    # results = common_token_model.inference_dataset(dataset_dict["dev"], top_k=100)
    # evaluator = PerformanceEvaluator()
    # evaluator.calculate_classification_performance(
    #     dataset_dict["dev"]["label"], results["preds"]
    # )
    # results = common_token_model.inference_dataset(dataset_dict["train"], top_k=1500)
    # evaluator = PerformanceEvaluator()
    # evaluator.calculate_classification_performance(
    #     dataset_dict["train"]["label"], results["preds"]
    # )

    # tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # text = "本论文介绍了一种专用于大棚蔬菜移栽的机械装置的设计，这种装置旨在提高移栽效率和确保幼苗成活率。在本研究中，我们首先讨论了现有大棚移栽技术的局限性，然后介绍了新设计的移栽机的主要结构和功能原理。该装置采用模块化设计，包含自动化控制系统、精准深度控制机构和苗木保护机制，能够适应不同类型的蔬菜移栽需求。实验结果表明，与传统人工移栽方法相比，该移栽机可以显著节省人力成本并提高作业效率，同时减少了对幼苗的损伤。最后，通过田间试验验证，本文所设计的移栽机表现出良好的适应性和可靠性，具有广阔的应用前景。本研究还对不同参数设置下的移栽性能进行了详细分析，包括移栽速度、深度控制精度和苗木保护效果。结果显示，该装置在不同蔬菜品种和土壤条件下均能保持一致的高性能输出。此外，我们对系统的自动化控制部分进行了敏感性分析，以确保其在各种环境变化和故障情况下的稳定操作。通过与实际用户的反馈和反复测试，进一步优化了装置的用户界面和操作流程，使其更便于使用。同时，我们探讨了采用新兴传感技术和AI算法进行未来升级的可能性，以实现更智能化的操作和进一步提升移栽效率。综上所述，该移栽机不仅能显著提高大棚蔬菜移栽的作业效率，还为农业机械化和智能化提供了一个具有应用潜力的解决方案。未来的研究将着重于优化装置的能耗和成本效益，以满足更广范围内的农业需求。"
    # token_ids = tokenizer.encode(text)  # 转换为 token ID
    # decoded_tokens = [tokenizer.decode([tid]) for tid in token_ids]
    # print(decoded_tokens)

    common_token_model.draw_plot(
        dataset_dict["dev"],
        dataset_dict["dev"]["label"],
        start_topk=100,
        end_topk=2000,
        step=100,
    )
