# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class Config:
    """项目全局路径配置"""

    PROJECT_DIR = "/data/wangzhuoshang/project/mgtd-sys"

    DATA_DIR = os.path.join(PROJECT_DIR, "data")

    RAW_DATA = {
        "train": os.path.join(DATA_DIR, "data_raw/train.json"),
        "dev": os.path.join(DATA_DIR, "data_raw/dev.json"),
        "test": os.path.join(DATA_DIR, "data_raw/test.json"),
    }

    SFT_DATA = {
        "alpaca_format": {
            "train": os.path.join(DATA_DIR, "data_sft/nlpcc25_task1_train.json"),
            "dev": os.path.join(DATA_DIR, "data_sft/nlpcc25_task1_dev.json"),
        }
    }

    DETECTOR_DIR = os.path.join(PROJECT_DIR, "detector")

    CKPT_DIR = os.path.join(PROJECT_DIR, "detector/ckpt")

    CACHE_DIR = os.path.join(PROJECT_DIR, "detector/cache")

    LOG_DIR = os.path.join(PROJECT_DIR, "detector/log")

    @staticmethod
    def prepare_models_config(**kwargs) -> list[dict]:
        """集成模型参数配置
        - `id`：模型代号
        - `cls`：模型所属类
        - `init_args`：模型的初始化参数
        - `init_kwargs`：模型的初始化关键字参数
        - `args`：模型推理的参数
        - `kwargs`：模型推理的关键字参数
        - `determinative`：该模型是否是“决定性”指标
            - 值为1表示具有机生文本的“一票赞成权”或人类文本的“一票否决权”
            - 值为0表示具有机生文本的“一票否决权”或人类文本的“一票赞成权”
            - 值为-1表示非确定性指标
        - `vote`：该模型是否参与集成投票
            - 值为True表示参与投票
            - 值为False表示不参与投票
        - `importance`：该模型的“重要性”（越重要的模型在加权投票时会被赋予更高的权重）
        - `active`：该模型是否激活（不激活则不进行推理及投票）
            - 值为True表示该模型激活
            - 值为False表示该模型宕机
        """
        # pylint: disable=import-outside-toplevel
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
        )
        from detector.model.neural_based.chinese_bert_detector import (
            ChineseBertForAIGTDetection,
        )
        from detector.model.neural_based.llm_detector import (
            GLM4ForAIGTDetection,
            Qwen25ForAIGTDetection,
        )
        from detector.model.neural_based.fast_detectgpt import FastDetectGPT
        from detector.model.neural_based.binoculars import BinocularsDetector
        from detector.model.rule_based.special_token import SpecialTokenModel
        from detector.model.rule_based.sentence_segment import SentenceSegmentModel
        from detector.model.rule_based.consecutive_punctuation import (
            ConsecutivePunctuationModel,
        )
        from detector.model.rule_based.common_phrase import CommonPhraseModel
        from detector.model.rule_based.common_token import CommonTokenModel

        models_config = [
            {
                "id": "neural-01",
                "note": "全量微调的chinese-roberta模型",
                "cls": ChineseBertForAIGTDetection,
                "init_args": [],
                "init_kwargs": {
                    "model_path": os.path.join(
                        Config.CKPT_DIR, "chinese-roberta-sft_03-21_16-39_938"
                    ),  # "chinese-roberta-sft_03-21_14-11_926"
                    "tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "chinese-roberta-wwm-ext-large"
                    ),
                },
                "args": [],
                "kwargs": {
                    "device": "cuda",
                },
                "determinative": -1,
                "vote": True,
                "importance": 50,
                "active": True,
                "for_joint": True,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "neural-02",
                "note": "全量微调的chinese-bert模型",
                "cls": ChineseBertForAIGTDetection,
                "init_args": [],
                "init_kwargs": {
                    "model_path": os.path.join(
                        Config.CKPT_DIR, "chinese-bert-sft_03-28_20-17_908"
                    ),
                    "tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "chinese-bert-wwm-ext"
                    ),
                },
                "args": [],
                "kwargs": {
                    "device": "cuda",
                },
                "determinative": -1,
                "vote": True,
                "importance": 60,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "neural-03",
                "note": "lora微调的glm4-chat模型",
                "cls": GLM4ForAIGTDetection,
                "init_args": [],
                "init_kwargs": {
                    "base_model_path": os.path.join(Config.CKPT_DIR, "glm-4-9b-chat"),
                    "lora_sft_path": os.path.join(
                        Config.CKPT_DIR, "glm-4-9b-chat-lora/checkpoint-2000"
                    ),
                    "base_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "glm-4-9b-chat"
                    ),
                    "base_model_cls": AutoModelForCausalLM,
                    "base_tokenizer_cls": AutoTokenizer,
                },
                "args": [],
                "kwargs": {
                    "gen_kwargs": {
                        "max_new_tokens": 32,
                        "do_sample": False,
                    },
                    "batch_size": 4,
                    "add_default_prompt": True,
                    "add_default_system_prompt": False,  # 依照lora-sft时的设置
                    "ignore_double_newline": True,
                    "device": "cuda",
                },
                "determinative": -1,
                "vote": True,
                "importance": 85,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "neural-04",
                "note": "lora微调的qwen2.5-Instruct模型",
                "cls": Qwen25ForAIGTDetection,
                "init_args": [],
                "init_kwargs": {
                    "base_model_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "lora_sft_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct-lora-2/checkpoint-2000"
                    ),
                    "base_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "base_model_cls": AutoModelForCausalLM,
                    "base_tokenizer_cls": AutoTokenizer,
                },
                "args": [],
                "kwargs": {
                    "gen_kwargs": {
                        "max_new_tokens": 32,
                        "do_sample": False,
                    },
                    "batch_size": 1,
                    "add_default_prompt": True,
                    "add_default_system_prompt": True,  # 依照lora-sft时的设置
                    "ignore_double_newline": True,
                    "device": "cuda",
                },
                "determinative": -1,
                "vote": True,
                "importance": 1000,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": True,
            },
            {
                "id": "neural-05",
                "note": "lora微调的qwen2.5-Instruct模型（超短文本）",
                "cls": Qwen25ForAIGTDetection,
                "init_args": [],
                "init_kwargs": {
                    "base_model_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "lora_sft_path": os.path.join(
                        Config.CKPT_DIR,
                        "Qwen2.5-7B-Instruct-lora-extreme_short/checkpoint-1250",
                    ),
                    "base_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "base_model_cls": AutoModelForCausalLM,
                    "base_tokenizer_cls": AutoTokenizer,
                },
                "args": [],
                "kwargs": {
                    "gen_kwargs": {
                        "max_new_tokens": 32,
                        "do_sample": False,
                    },
                    "batch_size": 1,
                    "add_default_prompt": True,
                    "add_default_system_prompt": True,  # 依照lora-sft时的设置
                    "ignore_double_newline": True,
                    "device": "cuda",
                },
                "determinative": -1,
                "vote": True,
                "importance": 40,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "neural-06",
                "note": "lora微调的qwen2.5-Instruct模型（短文本）",
                "cls": Qwen25ForAIGTDetection,
                "init_args": [],
                "init_kwargs": {
                    "base_model_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "lora_sft_path": os.path.join(
                        Config.CKPT_DIR,
                        "Qwen2.5-7B-Instruct-lora-short/checkpoint-1250",
                    ),
                    "base_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "base_model_cls": AutoModelForCausalLM,
                    "base_tokenizer_cls": AutoTokenizer,
                },
                "args": [],
                "kwargs": {
                    "gen_kwargs": {
                        "max_new_tokens": 32,
                        "do_sample": False,
                    },
                    "batch_size": 1,
                    "add_default_prompt": True,
                    "add_default_system_prompt": True,  # 依照lora-sft时的设置
                    "ignore_double_newline": True,
                    "device": "cuda",
                },
                "determinative": -1,
                "vote": True,
                "importance": 60,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "neural-07",
                "note": "Fast-DetectGPT",
                "cls": FastDetectGPT,
                "init_args": [],
                "init_kwargs": {
                    "sampling_model_path": os.path.join(Config.CKPT_DIR, "Qwen2.5-7B"),
                    "sampling_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B"
                    ),
                    "scoring_model_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "scoring_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                },
                "args": [],
                "kwargs": {
                    "analytical": False,
                    "num_samples": 10000,
                    "ignore_double_newline": True,
                    "threshold": 1.070897,
                    "fine_grained_threshold": True,
                    "auto_compute_threshold": False,
                    "labels": kwargs["labels"],
                    "device_sampling": "cuda:0",
                    "device_scoring": "cuda:1",
                },
                "determinative": -1,
                "vote": True,
                "importance": 70,
                "active": True,
                "for_joint": True,
                "for_joint_extreme_short": True,
                "strong": False,
            },
            {
                "id": "neural-08",
                "note": "Fast-DetectGPT-analytical",
                "cls": FastDetectGPT,
                "init_args": [],
                "init_kwargs": {
                    "sampling_model_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "sampling_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "scoring_model_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "scoring_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                },
                "args": [],
                "kwargs": {
                    "analytical": True,
                    "num_samples": 10000,
                    "ignore_double_newline": True,
                    "threshold": -0.078113,
                    "fine_grained_threshold": True,
                    "auto_compute_threshold": False,
                    "labels": kwargs["labels"],
                    "device_sampling": "cuda:0",
                    "device_scoring": "cuda:1",
                },
                "determinative": -1,
                "vote": True,
                "importance": 70,
                "active": True,
                "for_joint": True,
                "for_joint_extreme_short": True,
                "strong": False,
            },
            {
                "id": "neural-09",
                "note": "Fast-DetectGPT-analytical",
                "cls": FastDetectGPT,
                "init_args": [],
                "init_kwargs": {
                    "sampling_model_path": os.path.join(
                        Config.CKPT_DIR, "glm-4-9b-chat"
                    ),
                    "sampling_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "glm-4-9b-chat"
                    ),
                    "scoring_model_path": os.path.join(
                        Config.CKPT_DIR, "glm-4-9b-chat"
                    ),
                    "scoring_tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "glm-4-9b-chat"
                    ),
                },
                "args": [],
                "kwargs": {
                    "analytical": True,
                    "num_samples": 10000,
                    "ignore_double_newline": True,
                    "threshold": -5.656250,
                    "fine_grained_threshold": True,
                    "auto_compute_threshold": False,
                    "labels": kwargs["labels"],
                    "device_sampling": "cuda:0",
                    "device_scoring": "cuda:1",
                },
                "determinative": -1,
                "vote": True,
                "importance": 70,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "neural-10",
                "note": "Binoculars",
                "cls": BinocularsDetector,
                "init_args": [],
                "init_kwargs": {
                    "observer_model_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                    "performer_model_path": os.path.join(Config.CKPT_DIR, "Qwen2.5-7B"),
                    "tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ),
                },
                "args": [],
                "kwargs": {
                    "ignore_double_newline": True,
                    "threshold": 0.943096,
                    "fine_grained_threshold": True,
                    "auto_compute_threshold": False,
                    "labels": kwargs["labels"],
                    "device_observer": "cuda:0",
                    "device_performer": "cuda:1",
                },
                "determinative": -1,
                "vote": True,
                "importance": 75,
                "active": True,
                "for_joint": True,
                "for_joint_extreme_short": True,
                "strong": False,
            },
            {
                "id": "neural-11",
                "note": "全量微调的chinese-roberta模型",
                "cls": ChineseBertForAIGTDetection,
                "init_args": [],
                "init_kwargs": {
                    "model_path": os.path.join(
                        Config.CKPT_DIR, "chinese-bert-sft_04-18_12-42_809"
                    ),
                    "tokenizer_path": os.path.join(
                        Config.CKPT_DIR, "chinese-roberta-wwm-ext-large"
                    ),
                },
                "args": [],
                "kwargs": {
                    "device": "cuda",
                },
                "determinative": -1,
                "vote": True,
                "importance": 0,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": True,
                "strong": False,
            },
            {
                "id": "rule-01",
                "note": "特殊标记规则",
                "cls": SpecialTokenModel,
                "init_args": [],
                "init_kwargs": {},
                "args": [],
                "kwargs": {},
                "determinative": 1,
                "vote": False,  # 建议不参与投票
                "importance": 9999,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "rule-02",
                "note": "子句规则",
                "cls": SentenceSegmentModel,
                "init_args": [],
                "init_kwargs": {
                    "decision_threshold": 10,
                },
                "args": [],
                "kwargs": {},
                "determinative": -1,
                "vote": True,  # 视情况参与投票
                "importance": 10,
                "active": True,
                "for_joint": True,
                "for_joint_extreme_short": True,
                "strong": False,
            },
            {
                "id": "rule-03",
                "note": "连续标点规则",
                "cls": ConsecutivePunctuationModel,
                "init_args": [],
                "init_kwargs": {},
                "args": [],
                "kwargs": {},
                "determinative": 0,
                "vote": False,  # 建议不参与投票
                "importance": 9999,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "rule-04",
                "note": "常见短语规则",
                "cls": CommonPhraseModel,
                "init_args": [],
                "init_kwargs": {},
                "args": [],
                "kwargs": {},
                "determinative": -1,
                "vote": True,
                "importance": 5,
                "active": True,  # 建议关闭
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "rule-05",
                "note": "常见token规则",
                "cls": CommonTokenModel,
                "init_args": [],
                "init_kwargs": {
                    "tokenizer_path": os.path.join(Config.CKPT_DIR, "glm-4-9b-chat"),
                    "build_token_dict": True,
                    "refer_dataset": kwargs["refer_dataset"],
                },
                "args": [],
                "kwargs": {
                    "top_k": 1500,
                },
                "determinative": -1,
                "vote": True,
                "importance": 40,
                "active": True,
                "for_joint": True,
                "for_joint_extreme_short": True,
                "strong": False,
            },
        ]

        Config.check_models_config_validity(models_config)

        return models_config

    @staticmethod
    def prepare_models_config_joint(models_config: list[dict], **kwargs) -> list[dict]:
        # pylint: disable=import-outside-toplevel
        from detector.model.neural_based.joint_features import JointFeaturesMLP

        models_config_joint = [
            {
                "id": "joint-01",
                "note": "混合特征模型",
                "cls": JointFeaturesMLP,
                "init_args": [],
                "init_kwargs": {
                    "model_path": os.path.join(
                        Config.CKPT_DIR, "joint-features-mlp_04-13_15-51_951.pth"
                    ),
                    "models_config": models_config,
                    "feat_dim": 1035,
                    "hidden_dim": 512,
                },
                "args": [],
                "kwargs": {
                    "dataset_flag": kwargs["dataset_flag"],
                },
                "determinative": -1,
                "vote": True,
                "importance": 80,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
            {
                "id": "joint-02",
                "note": "混合特征模型",
                "cls": JointFeaturesMLP,
                "init_args": [],
                "init_kwargs": {
                    "model_path": os.path.join(
                        Config.CKPT_DIR, "joint-features-mlp_04-18_18-12_842.pth"
                    ),
                    "models_config": models_config,
                    "feat_dim": 1035,
                    "hidden_dim": 512,
                },
                "args": [],
                "kwargs": {
                    "dataset_flag": kwargs["dataset_flag"],
                },
                "determinative": -1,
                "vote": True,
                "importance": 0,
                "active": True,
                "for_joint": False,
                "for_joint_extreme_short": False,
                "strong": False,
            },
        ]

        Config.check_models_config_validity(models_config_joint)

        return models_config_joint

    @staticmethod
    def check_models_config_validity(models_config: list[dict]):
        """检查模型配置合法性"""
        logging.info("Checking for models config validity ...")
        ids = set()
        for model_config in models_config:
            ids.add(model_config["id"])
            assert model_config["determinative"] in [
                -1,
                0,
                1,
            ], f"Bad config for {model_config["determinative"]=}"
            assert isinstance(
                model_config["vote"], bool
            ), f"Bad config for {model_config["vote"]=}"
            assert (
                isinstance(model_config["importance"], int)
                and model_config["importance"] >= 0
            ), f"Bad config for {model_config["importance"]=}"
            assert isinstance(
                model_config["active"], bool
            ), f"Bad config for {model_config["active"]=}"
            assert isinstance(
                model_config["for_joint"], bool
            ), f"Bad config for {model_config["for_joint"]=}"
            assert isinstance(
                model_config["for_joint_extreme_short"], bool
            ), f"Bad config for {model_config["for_joint_extreme_short"]=}"
        assert len(ids) == len(models_config), "Duplicate model ids"
