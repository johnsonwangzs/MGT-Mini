# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os
import logging
import torch
import pickle
import statistics
import numpy as np
from scipy import stats
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from detector.config import Config
from detector.eval.evaluator import PerformanceEvaluator
from detector.utils import generate_cache_path
from detector.model.neural_based.base_neural_model import BaseNeuralModel

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FastDetectGPT(BaseNeuralModel):

    desc = repr("Zero-shot detector Fast-DetectGPT")

    def __init__(
        self,
        sampling_model_path: str,
        sampling_tokenizer_path: str,
        scoring_model_path: str,
        scoring_tokenizer_path: str,
    ):
        super().__init__()
        self.sampling_model_path = sampling_model_path
        self.sampling_tokenizer_path = sampling_tokenizer_path
        self.scoring_model_path = scoring_model_path
        self.scoring_tokenizer_path = scoring_tokenizer_path

        # 默认判定阈值 (数据集_采样_评分_模式_阈值)
        self.threshold = {
            "glm4-9b-chat_glm4-9b-chat": -4.32,
            "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1": -1.311762,
            "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1": 1.025630,
            "devset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1": -0.078113,
            "devset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1": -5.656250,
            "devset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_median": 0.342083,
            "devset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1": 1.070897,
            "devset_qwen2.5-7b_qwen2.5-7b-instruct_normal_median": 1.019218,
            "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_64": 0.707698,
            "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_128": 0.692254,
            "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_256": 0.968081,
            "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_512": 1.139770,
            "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_other": 0.692545,
            "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_64": -0.236470,
            "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_128": 0.071417,
            "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_256": 0.023761,
            "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_512": 0.210341,
            "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_other": -0.031909,
            "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_64": -1.578125,
            "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_128": -1.5625,
            "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_256": -2.609375,
            "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_512": -3.53125,
            "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_other": -6.21875,
        }

    def _load_model(self):
        # pylint: disable=attribute-defined-outside-init
        logging.info("Loading sampling model: %s", self.sampling_model_path)
        self.sampling_model = AutoModelForCausalLM.from_pretrained(
            self.sampling_model_path, trust_remote_code=True
        )
        self.sampling_tokenizer = AutoTokenizer.from_pretrained(
            self.sampling_tokenizer_path, trust_remote_code=True
        )
        logging.info("Loading scoring model: %s", self.scoring_model_path)
        self.scoring_model = AutoModelForCausalLM.from_pretrained(
            self.scoring_model_path, trust_remote_code=True
        )
        self.scoring_tokenizer = AutoTokenizer.from_pretrained(
            self.scoring_tokenizer_path, trust_remote_code=True
        )

    def inference_example(
        self,
        text: str,
        analytical: bool = True,
        num_samples: int = 10000,
        ignore_double_newline: bool = True,
        threshold: float = 0.0,
        device_sampling="cpu",
        device_scoring="cpu",
        **kwargs,
    ):

        logging.info("Performing inference using %s ...", self.__class__.__name__)
        self._load_model()
        if ignore_double_newline is True:
            text = text.replace("\n\n", "")

        if analytical is True:
            logging.info("Set Fast-DetectGPT mode to 'analytical'.")
            curvature_score = self._inference_analytical(text, device=device_sampling)

        else:
            samples, inputs = self._sample_alternative(
                text, num_samples=num_samples, device=device_sampling
            )
            log_p_x_given_x, alternative_log_p_x_given_x = (
                self._calculate_conditional_probability(
                    samples, inputs, device=device_scoring
                )
            )
            curvature_score = self._evaluate_curvature(
                log_p_x_given_x, alternative_log_p_x_given_x
            )

        print(curvature_score)
        return curvature_score

    def inference_dataset(
        self,
        dataset: Dataset,
        analytical: bool = True,
        num_samples: int = 10000,
        ignore_double_newline: bool = True,
        threshold: float = 0.0,
        fine_grained_threshold: bool = True,
        auto_compute_threshold: bool = True,
        threshold_compute_method: str = "f1",
        labels: list = None,
        device_sampling="cpu",
        device_scoring="cpu",
        **kwargs,
    ):

        logging.info("Performing inference using %s ...", self.__class__.__name__)

        if analytical is True:
            logging.info("Set Fast-DetectGPT mode to 'analytical'.")

        preds_cache_path = generate_cache_path(
            data=dataset,
            filename=f"{self.__class__.__name__}.tmp",
            analytical=analytical,
            num_samples=num_samples,
            ignore_double_newline=ignore_double_newline,
            sampling_model=self.sampling_model_path,
            scoring_model=self.scoring_model_path,
        )
        if not os.path.exists(preds_cache_path):
            logging.info("Using device %s, %s", device_sampling, device_scoring)
            self._load_model()
            curvature_scores = []
            for example in tqdm(dataset["text"], desc="Inferring ..."):
                if ignore_double_newline is True:
                    example = example.replace("\n\n", "")

                if analytical is True:
                    curvature_score = self._inference_analytical(
                        example, device=device_sampling
                    )
                else:
                    samples, inputs = self._sample_alternative(
                        example, num_samples=num_samples, device=device_sampling
                    )
                    log_p_x_given_x, alternative_log_p_x_given_x = (
                        self._calculate_conditional_probability(
                            samples, inputs, device=device_scoring
                        )
                    )
                    curvature_score = self._evaluate_curvature(
                        log_p_x_given_x, alternative_log_p_x_given_x
                    )
                curvature_scores.append(curvature_score)

            cache_data = {"curvature_scores": curvature_scores}
            with open(preds_cache_path, "wb") as f_cache:
                pickle.dump(cache_data, f_cache)
                logging.info(
                    "Save model prediction results to cache `%s`", preds_cache_path
                )

        else:
            logging.info("Reading prediction results from cache `%s`", preds_cache_path)
            with open(preds_cache_path, "rb") as f_cache:
                cache_data = pickle.load(f_cache)
            curvature_scores = cache_data["curvature_scores"]

        # self._show_stat(curvature_scores)

        if auto_compute_threshold is True:
            assert labels is not None, "Labels are required to calculate the threshold."
            threshold = self.compute_threshold(
                curvature_scores, labels, method=threshold_compute_method
            )
            logging.info(
                "Set threshold to %f (based on %s)", threshold, threshold_compute_method
            )
        elif threshold is not None:
            logging.info("Set threshold to %f", threshold)

        preds = self.make_prediction(
            curvature_scores,
            analytical=analytical,
            threshold=threshold,
            threshold_compute_method=threshold_compute_method,
            fine_grained_threshold=fine_grained_threshold,
            dataset=dataset,
        )

        return {"preds": preds, "curvature_scores": curvature_scores}

    @staticmethod
    def _show_stat(data: list):

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

    def compute_threshold(
        self, curvature_scores: list, labels: list, method: str = "f1"
    ):

        assert method in ["f1", "median"], "Invalid compute method"

        if method == "median":
            human_scores, machine_scores = [], []
            for i, score in enumerate(curvature_scores):
                if labels[i] == 0:
                    human_scores.append(score)
                else:
                    machine_scores.append(score)

            human_median = statistics.median(human_scores)
            machine_median = statistics.median(machine_scores)

            threshold = (human_median + machine_median) / 2

        else:
            threshold, f1 = PerformanceEvaluator.find_best_threshold_f1(
                y_true=labels, y_scores=curvature_scores
            )

        return threshold

    def make_prediction(
        self,
        curvature_scores: list,
        analytical: bool,
        threshold: float,
        threshold_compute_method: str,
        fine_grained_threshold: bool,
        dataset: Dataset = None,
    ):

        preds = []
        if fine_grained_threshold is False:
            for score in curvature_scores:
                preds.append(1 if score > threshold else 0)
        else:
            assert (
                dataset is not None
            ), "Require dataset to apply fine-grained threshold"
            for score, example in zip(curvature_scores, dataset["text"]):
                len_text = len(example)
                if analytical is False:
                    if self.sampling_model_path == os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B"
                    ) and self.scoring_model_path == os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ):
                        if threshold_compute_method == "f1":
                            if len_text < 75:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_64"
                                    ]
                                    else 0
                                )
                            elif len_text < 150:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_128"
                                    ]
                                    else 0
                                )
                            elif len_text < 300:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_256"
                                    ]
                                    else 0
                                )
                            elif len_text < 600:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_512"
                                    ]
                                    else 0
                                )
                            else:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b_qwen2.5-7b-instruct_normal_f1_other"
                                    ]
                                    else 0
                                )
                else:
                    if self.sampling_model_path == os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ) and self.scoring_model_path == os.path.join(
                        Config.CKPT_DIR, "Qwen2.5-7B-Instruct"
                    ):
                        if threshold_compute_method == "f1":
                            if len_text < 75:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_64"
                                    ]
                                    else 0
                                )
                            elif len_text < 150:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_128"
                                    ]
                                    else 0
                                )
                            elif len_text < 300:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_256"
                                    ]
                                    else 0
                                )
                            elif len_text < 600:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_512"
                                    ]
                                    else 0
                                )
                            else:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_qwen2.5-7b-instruct_qwen2.5-7b-instruct_analytical_f1_other"
                                    ]
                                    else 0
                                )
                    elif self.sampling_model_path == os.path.join(
                        Config.CKPT_DIR, "glm-4-9b-chat"
                    ) and self.scoring_model_path == os.path.join(
                        Config.CKPT_DIR, "glm-4-9b-chat"
                    ):
                        if threshold_compute_method == "f1":
                            if len_text < 75:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_64"
                                    ]
                                    else 0
                                )
                            elif len_text < 150:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_128"
                                    ]
                                    else 0
                                )
                            elif len_text < 300:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_256"
                                    ]
                                    else 0
                                )
                            elif len_text < 600:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_512"
                                    ]
                                    else 0
                                )
                            else:
                                preds.append(
                                    1
                                    if score
                                    > self.threshold[
                                        "trainset_glm-4-9b-chat_glm-4-9b-chat_analytical_f1_other"
                                    ]
                                    else 0
                                )

        return preds

    def _inference_analytical(self, text: str, device="cpu"):

        self.sampling_model.to(device)
        self.sampling_model.eval()

        # 分词
        inputs = self.sampling_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=900,  # 这里需要限制长度，否则V100 32G可能会爆显存
            add_special_tokens=False,
        ).to(device)
        input_ids = inputs["input_ids"]  # shape: [1, seq_len]

        # 获取logits
        with torch.no_grad():
            outputs = self.sampling_model(**inputs)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        # tokens_list = self.sampling_tokenizer.convert_ids_to_tokens(input_ids[0])
        # tokens_decoded = [
        #     t.decode("utf-8") if isinstance(t, bytes) else t for t in tokens_list
        # ]
        # print(tokens_decoded)

        # 做偏移
        shift_logits = logits[:, :-1, :]
        shift_input_ids = input_ids[:, 1:]  # 从第二个token开始

        # 计算原文的log prob (log p(x|x))
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_input_ids.unsqueeze(-1)).squeeze(
            -1
        )  # shape: [1, seq_len-1]
        log_p_x_given_x = token_log_probs.sum()

        # 计算log prob的期望（等于负熵）
        probs = torch.softmax(shift_logits, dim=-1)  # shape: [1, seq_len-1, vocab_size]
        entropy = -torch.sum(probs * log_probs, dim=-1).squeeze(0)  # shape: [seq_len-1]
        mu_tilde = -entropy.sum()

        # 计算方差 (σ²) = E[log p²] - (E[log p])²
        log_probs_sq = log_probs**2
        exp_log_probs_sq = torch.sum(probs * log_probs_sq, dim=-1).squeeze(
            0
        )  # shape: [seq_len-1]
        sigma2_tilde = (exp_log_probs_sq - entropy**2).sum()
        sigma_tilde = torch.sqrt(sigma2_tilde)

        # 计算曲率分数
        curvature_score = ((log_p_x_given_x - mu_tilde) / sigma_tilde).item()

        return curvature_score

    def _sample_alternative(self, text: str, num_samples: int = 10000, device="cpu"):

        logging.debug("Sampling alternatives ...")
        logging.debug("Set num_samples to %d", num_samples)
        self.sampling_model.to(device)
        self.sampling_model.eval()

        # 分词
        inputs = self.sampling_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=500,  # 这里需要限制长度，否则V100 32G可能会爆显存
            add_special_tokens=False,
        ).to(device)

        # 获取logits
        with torch.no_grad():
            outputs = self.sampling_model(**inputs)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        shift_logits = logits[:, :-1, :]  # shape: [1, seq_len-1, vocab_size]

        # 计算每个 token 的 log-probability
        log_probs = torch.log_softmax(
            shift_logits, dim=-1
        )  # shape: [1, seq_len-1, vocab_size]

        # 采样
        samples = torch.distributions.categorical.Categorical(logits=log_probs).sample(
            [num_samples]
        )  # shape: [num_samples, 1, seq_len-1]

        return samples, inputs

    def _calculate_conditional_probability(
        self,
        samples: torch.Tensor,
        inputs: dict,
        device="cpu",
    ):

        logging.debug("Calculating conditional probability ...")
        self.scoring_model.to(device)
        self.scoring_model.eval()

        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]

        # 获取logits
        with torch.no_grad():
            outputs = self.scoring_model(**inputs)
            logits = outputs.logits  # shape: [1, seq_len, vocab_size]

        shift_logits = logits[:, :-1, :]  # shape: [1, seq_len-1, vocab_size]
        shift_input_ids = input_ids[:, 1:]  # 从第二个token开始 shape: [1, seq_len-1]

        # 计算每个 token 的 log-probability
        log_probs = torch.log_softmax(
            shift_logits, dim=-1
        )  # shape: [1, seq_len-1, vocab_size]

        # 取出每个token的log cond-prob
        token_log_probs = log_probs.gather(2, shift_input_ids.unsqueeze(-1)).squeeze(
            -1
        )  # shape: [1, seq_len-1]

        # 计算原文x的log prob：log p(x|x) = log p(x) = sum(log p(x_j|x_<j))
        log_p_x_given_x = token_log_probs.sum()

        # 并行计算所有samples的log prob
        # log_probs: [1, seq_len-1, vocab_size]
        # samples: [num_samples, 1, seq_len-1] → 需要调整形状以匹配 log_probs
        samples = samples.to(device)
        num_samples = samples.shape[0]
        expanded_log_probs = log_probs.expand(
            num_samples, -1, -1
        )  # [num_samples, seq_len-1, vocab_size]
        samples = samples.squeeze(1)  # [num_samples, seq_len-1]

        # 取出每个sample的每个token的log-probs
        alternative_token_log_probs = expanded_log_probs.gather(
            2, samples.unsqueeze(-1)
        ).squeeze(
            -1
        )  # [num_samples, seq_len-1]

        # 计算每个sample的log prob：log p(x') = sum(log p(x'_j|x_<j))
        alternative_log_p_x_given_x = alternative_token_log_probs.sum(
            dim=1
        )  # [num_samples]

        return log_p_x_given_x, alternative_log_p_x_given_x

    def _evaluate_curvature(
        self,
        log_p_x_given_x: torch.Tensor,
        alternative_log_p_x_given_x: torch.Tensor,
    ):

        logging.debug("Evaluating curvature score ...")
        # 计算均值和标准差
        mu_tilde = alternative_log_p_x_given_x.mean()  # 均值 μ
        sigma_tilde = alternative_log_p_x_given_x.std(
            unbiased=False
        )  # 标准差 σ（使用有偏估计）

        # 防止除以零
        if sigma_tilde == 0:
            return torch.tensor(0.0, device=log_p_x_given_x.device)

        # 计算曲率分数
        curvature = (log_p_x_given_x - mu_tilde) / sigma_tilde
        return curvature.detach().item()


if __name__ == "__main__":

    sampling_model_path = os.path.join(Config.CKPT_DIR, "Qwen2.5-7B")
    scoring_model_path = os.path.join(Config.CKPT_DIR, "Qwen2.5-7B-Instruct")
    detector = FastDetectGPT(
        sampling_model_path=sampling_model_path,
        sampling_tokenizer_path=sampling_model_path,
        scoring_model_path=scoring_model_path,
        scoring_tokenizer_path=scoring_model_path,
    )

    # 机器测试样本
    # test_text = "同学是靖江的，推荐了我们吃这家羊肉，店面很大，大堂明亮，关键是不会有很重羊膻味儿，这一点真的让我很惊喜。很多地方的羊肉店一进门就能闻到浓重的羊膻味，而这家店则完全没有这种困扰。羊肉煮得非常软烂，入口即化，调味也恰到好处，不会掩盖住羊肉自身的鲜美。我们点了几道招牌菜，包括羊肉火锅、红烧羊肉和孜然羊排，味道都非常好。火锅的羊肉切片均匀，煮熟后口感极佳，配上清新的蔬菜和独特的蘸酱，真是让人食欲大开。红烧羊肉色泽诱人，汤汁浓稠，非常入味，和米饭是绝配。孜然羊排外焦里嫩，香气扑鼻，配上一杯清凉饮料，实在是完美的享受。服务方面，服务员都很热情，点菜时会耐心介绍每一道菜的特色，还贴心地帮我们推荐了几道适合夏季的清爽小菜。总体来说，这次的用餐体验非常愉快，既满足了口腹之欲，也感受到了靖江当地人的热情好客。如果以后有机会，一定还会再来光顾，也会带上更多朋友一起来分享这份美味。推荐给所有喜欢羊肉的朋友们，这真的是一家不可错过的餐厅！"
    # test_text = "经过一顿美妙的火锅体验，我完全赞同这一说法。川骄的火锅确实名不虚传，店内的环境布置典雅，给人一种舒适放松的氛围。而火锅的味道更是让人回味无穷，他们家的底料选材讲究，辣而不燥，麻而不苦，独特的秘制配方使得每一口都充满了层次感。涮菜的品种丰富，从新鲜的肉类到各种蔬菜，再到海鲜和豆制品，应有尽有，满足了我对火锅的所有期待。服务方面，工作人员态度热情，上菜迅速，让人倍感温馨。总之，川骄火锅绝对值得一试，下次聚餐我一定会再次光顾！在这个喧嚣的城市中，川骄火锅就像一个温馨的港湾，让人在享受美食的同时，还能暂时忘却外界的纷扰。我尤其喜欢他们家特色的蘸料，酸辣适中，既能提味又能去腻，完美地衬托出火锅的香浓。朋友们聚在一起，谈笑风生，火锅的热气蒸腾，仿佛时间都慢了下来。\n\n此外，川骄火锅的食材新鲜度也是让我印象深刻的一点。每一片肉都切得薄厚适中，涮个两三分钟便入口即化，而那些新鲜的蔬菜和海鲜，更是让人吃得放心。在这里，你不用担心食品安全的问题，每一道菜都让人吃得安心。\n\n当然，吃火锅不能只顾着吃，还要喝上几杯小酒。川骄的酒水种类繁多，从国产到进口，应有尽有，让人可以根据自己的口味选择。边涮火锅边品酒，真是人生一大乐事。\n\n最后，值得一提的是，川骄火锅的性价比非常高。虽然价格不算便宜，但考虑到这里的食材质量和服务态度，绝对是物有所值。下次带家人或者朋友来聚餐，我会毫不犹豫地推荐川骄火锅。相信在这里，我们都能找到属于自己的那份温暖和快乐。川骄，真是一家不可多得的火锅馆，值得大家一试再试！"
    # test_text = "这反映出江北万达作为城市繁华商圈的地标，人气旺盛，人气旺盛的背后，是这里的餐饮环境和美食质量受到了广大消费者的认可。无论是热闹的餐厅氛围，还是精致的美食菜品，都能让人流连忘返。当然，这也意味着就餐时间可能会相对紧张，建议大家提前查看菜单，选择适合自己的菜品，合理安排就餐时间，以确保能够享受到美食的同时，也能在舒适的环境中度过美好的用餐时光。总的来说，江北万达的东北风餐厅值得一来，无论是与朋友小聚，还是家庭聚餐，都能在此找到满意的美食和愉悦的氛围。江北万达的东北风餐厅，以其独特的地域特色和丰富的菜品选择，不仅满足了食客们的味蕾，更是成为了人们社交娱乐的新去处。在这里，你可以感受到浓厚的地方文化氛围，仿佛置身于那遥远的东北大地。\n\n餐厅的环境布置简洁大方，既体现了东北风格的粗犷豪迈，又不失温馨舒适。墙上挂满了东北特色的画作和剪纸，让人在用餐的同时，也能领略到浓厚的民俗风情。\n\n值得一提的是，江北万达的东北风餐厅在食材选择上极为讲究，严格把控质量，从源头确保食品安全。餐厅的厨师团队经验丰富，擅长运用东北传统烹饪技艺，将各种食材烹制得色香味俱佳。\n\n此外，餐厅的服务态度也值得称赞。服务员热情周到，对顾客的需求能够迅速响应，让顾客感受到家的温馨。在这里，你可以尽情享受美食，无需担心服务问题。\n\n当然，江北万达的东北风餐厅也有它的不足之处。由于人气旺盛，就餐高峰时段可能需要排队等候，这就需要大家提前规划好时间，以免耽误用餐。同时，餐厅的容量有限，在高峰时段可能会出现座位紧张的情况。\n\n总之，江北万达的东北风餐厅作为城市繁华商圈的地标，凭借其独特的魅力，赢得了广大消费者的喜爱。在这里，你不仅可以品尝到正宗的东北美食，还能感受到浓厚的文化氛围。无论是朋友聚会还是家庭聚餐，都是不错的选择。让我们共同期待江北万达的东北风餐厅在未来的发展中，为更多的食客带来美好的用餐体验。"
    # test_text = "不过，话说回来，这里的味道和服务真的是没得说。一进门就被温馨的氛围所吸引，装修风格简约而不失高雅，让人感觉宾至如归。\n\n菜品方面，海底捞的招牌火锅自然是无可挑剔，食材新鲜，调料丰富，可以根据个人口味进行调配。除了火锅，这里的其他菜品也颇具特色，比如川菜、粤菜、日料等，种类繁多，能够满足不同食客的需求。\n\n值得一提的是，海底捞的服务态度真的非常好。服务员态度亲切，主动帮客人夹菜、递调料，甚至还会在用餐过程中表演节目，让人感觉十分愉快。此外，餐厅还设有儿童游乐区，让带孩子的家长也能安心用餐。\n\n总的来说，虽然价格不菲，但海底捞的味道和服务确实值得这个价格。如果你是火锅爱好者，或者想要体验一下高端餐厅的氛围，不妨来海底捞一试。当然，现在优惠活动这么多，提前关注一下，就能享受到更实惠的价格了。总之，这是一次愉快的用餐体验，下次还会再来！"
    # test_text = "这种善良的本能，常常在不经意间改变着人与人之间的关系，甚至改变着命运。\n\n在一个偏远的小山村里，住着一位名叫李老汉的老人。他年过七旬，独自一人生活，靠种地为生。村里的人都知道，李老汉虽然生活清贫，但心地善良，总是乐于助人。无论是谁家遇到困难，他都会毫不犹豫地伸出援手。\n\n有一天，村里来了一个陌生的年轻人。他衣衫褴褛，面容憔悴，显然是经历了长途跋涉。年轻人名叫阿强，原本是城里的一名工人，但因为工厂倒闭，他失去了工作，又无处可去，只好四处流浪，最终来到了这个村子。\n\n阿强来到村口时，已经饿得几乎走不动了。他坐在路边，眼神空洞，仿佛对生活失去了希望。这时，李老汉正好从田里回来，看到了阿强。他走上前去，关切地问道：“小伙子，你这是怎么了？怎么坐在这里？”\n\n阿强抬起头，看到一位慈祥的老人，心中涌起一丝温暖。他低声说道：“老伯，我……我找不到工作，已经好几天没吃饭了。”\n\n李老汉听了，二话不说，拉起阿强的手说：“走，跟我回家，我给你弄点吃的。”\n\n阿强跟着李老汉回到了家。李老汉给他煮了一碗热腾腾的面条，还拿出自己珍藏的一点咸菜。阿强狼吞虎咽地吃完，感激地说道：“老伯，您真是我的救命恩人！我……我该怎么报答您？”\n\n李老汉笑了笑，摆摆手说：“不用报答。人都有困难的时候，能帮一把就帮一把。你接下来有什么打算？”\n\n阿强低下头，叹了口气：“我也不知道。我找不到工作，身上也没钱，不知道还能去哪里。”\n\n李老汉沉思片刻，说道：“这样吧，你先在我家住下，帮我干点农活。等你有机会了，再去找工作，怎么样？”\n\n阿强感激地点了点头：“老伯，您真是太好了！我一定会好好干的！”\n\n就这样，阿强在李老汉家住了下来。他每天早起晚归，帮李老汉种地、砍柴、挑水，干得十分卖力。李老汉看在眼里，心里也感到欣慰。他常常对阿强说：“年轻人，只要肯努力，日子总会好起来的。”\n\n几个月后，阿强在村里结识了一些朋友，听说附近的城市正在招工，工资待遇也不错。他决定去试一试。临走前，李老汉给他准备了一些干粮和路费，叮嘱道：“阿强，出门在外，要照顾好自己。遇到困难，别灰心，总会有人愿意帮你的。”\n\n阿强感动得热泪盈眶，紧紧握住李老汉的手：“老伯，您对我的恩情，我一辈子都不会忘记！等我有了出息，一定回来看您！”\n\n阿强离开后，李老汉的生活又恢复了平静。他依旧每天忙碌在田间地头，帮助村里需要帮助的人。虽然日子过得清贫，但他的心里却充满了满足和快乐。\n\n几年后的一天，村里突然来了一辆豪华的轿车。车上下来一位西装革履的年轻人，正是阿强。他如今已经成为一家大公司的经理，事业有成。他特意回到村里，想要报答李老汉当年的恩情。\n\n阿强找到李老汉时，老人正在田里干活。阿强走上前，激动地说道：“老伯，我回来了！我来看您了！”\n\n李老汉抬起头，看到阿强，脸上露出了欣慰的笑容：“阿强，你回来了！看你这样子，过得不错啊！”\n\n阿强点点头，说道：“老伯，当年要不是您收留我，给我饭吃，给我地方住，我可能早就饿死在路边了。今天，我是来报答您的！”\n\n李老汉摆摆手，笑道：“阿强，你不用报答我。当年帮你，是我心甘情愿的。看到你现在过得这么好，我就已经很开心了。”\n\n阿强坚持道：“老伯，您一定要接受我的报答！我已经在城里给您买了一套房子，您以后不用再这么辛苦了，跟我去城里享福吧！”\n\n李老汉摇了摇头，说道：“阿强，你的心意我领了。但我习惯了这里的生活，离不开这片土地。你还是把房子留给更需要的人吧。”\n\n阿强见李老汉态度坚决，只好作罢。但他并没有放弃，而是决定为村里做些事情。他出资修建了村里的学校，改善了村里的道路，还帮助许多贫困家庭解决了生活困难。村里的人都说，阿强是个知恩图报的好人。\n\n李老汉看着村里的变化，心里感到无比欣慰。他对阿强说：“阿强，你做得很好。帮助别人，不是为了得到回报，而是为了让这个世界变得更美好。”\n\n阿强点点头，说道：“老伯，您说得对。我会记住您的话，继续帮助那些需要帮助的人。”\n\n这个故事告诉我们，善良是一种无形的力量，它不仅能改变一个人的命运，还能传递温暖和希望。当我们看到别人陷于困境时，伸出援手，或许只是举手之劳，但却可能成为别人生命中的一盏明灯。而这份善意，最终也会以另一种方式回到我们身边，让世界变得更加美好。"
    # 人类测试样本
    # test_text = "同学是靖江的，推荐了我们吃这家羊肉，店面很大，大堂明亮，关键是不会有很重羊膻味儿…能看到后厨，大师傅们忙的热火朝天，后厨蛮干净的，印象分不错…总共三层楼，很大，包间很多，每一层有专门的服务员，点餐时有专业人员推荐，帮你搭配，很不错…羊汤，一大盆，半只羊排，很扎实啊，一块块肉都很大，而且嫩，羊汤很浓香…手拿骨头肯肉，怎一个爽字了得！！大蒜羊杂，符合苏南人口味，甜甜的，羊杂没有异味，口感特别好！！！只是大蒜没经过霜冻，有一点点老…砂锅羊血，赞一个，我的小伙伴一个人吃了半锅，脆嫩的口感，还有点像布丁，没有羊血的异味，调味适中，真的很好吃！色拉解腻，就是一盘草～也蛮好吃的还有他红烧的白丝鱼！很新鲜，入口即话，比我吃过清蒸的别有风味，里面的毛豆子很酥烂，淡淡的甜味…菠菜时新菜，都小小嫩嫩的，不会有涩的口号，补充维生素！！！最后，他家是烤全羊料理，还有很多其他的菜，推荐给大家………"
    # test_text = "【地址】世贸广场7楼，钟表后面的电梯可以直达。交通便利，地理位置好找。【环境】这家秀玉算是比较大的一家，可以选择自己喜欢的座位，环境不错，很适合聚会聊天。【服务】服务很热情，从一进门开始就有人主动询问，一路有人把你带到座位上。整个用餐过程的服务都做到及时，主动，热情，挺好的。【口味】连锁餐厅，口味有一定保障。虽然没有什么创新菜式，但是用餐，聚会，下午茶什么的没问题。老公喜欢这里的牛排，因为柔嫩分量足，吃得很饱；我喜欢这里的煲仔饭和泰皇炒饭，还有水果沙拉，吐司。吃完饭如果不想接着逛街，可以休息会，2点就有下午茶了，点上一壶茶，边喝边聊。唯一缺点就是世贸禁止明火，茶下面不能有蜡烛保温，这个天气，茶一会就冷了。但是我觉得可以用个插电的保温垫，服务是需要更细致，用心的！"

    score = detector.inference_example(
        test_text,
        analytical=False,
        num_samples=10000,
        ignore_double_newline=True,
        threshold=0.34,
        device_sampling=torch.device("cuda:5" if torch.cuda.is_available() else "cpu"),
        device_scoring=torch.device("cuda:6" if torch.cuda.is_available() else "cpu"),
    )

    # logging.info("Loading dataset ...")
    # trainset_path = os.path.join(Config.RAW_DATA["train"])
    # devset_path = os.path.join(Config.RAW_DATA["dev"])
    # dataset_dict = load_dataset(
    #     "json", data_files={"train": trainset_path, "dev": devset_path}
    # )
    # print(dataset_dict)
    # trainset = dataset_dict["train"]
    # train_labels = dataset_dict["train"]["label"]
    # devset = dataset_dict["dev"]
    # dev_labels = dataset_dict["dev"]["label"]

    # inferset = devset
    # infer_labels = dev_labels

    # results = detector.inference_dataset(
    #     inferset,
    #     analytical=True,
    #     num_samples=10000,
    #     ignore_double_newline=True,
    #     threshold=1.0245,
    #     auto_compute_threshold=True,
    #     labels=infer_labels,
    #     device_sampling=torch.device("cuda:2" if torch.cuda.is_available() else "cpu"),
    #     device_scoring=torch.device("cuda:4" if torch.cuda.is_available() else "cpu"),
    # )

    # evaluator = PerformanceEvaluator()
    # evaluator.calculate_classification_performance(
    #     infer_labels, results["preds"], scores=results["curvature_scores"]
    # )
    # evaluator.draw_roc(
    #     infer_labels, results["curvature_scores"], save_path="fast-detectgpt.png"
    # )
