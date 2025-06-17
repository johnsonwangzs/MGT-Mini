# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os.path
import logging
import pickle
import json
from datasets import load_dataset, Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from datasets import Dataset
from detector.config import Config
from detector.utils import generate_cache_path
from detector.eval.evaluator import PerformanceEvaluator
from detector.model.neural_based.base_neural_model import BaseNeuralModel

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Number of visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA logical device {i} -> {torch.cuda.get_device_name(i)}")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def prepare_data(
    dataset,
    models_config: list[dict],
    dataset_flag: str,
    labels: list = None,
):

    logging.info("Preparing data for dataset %s...", dataset_flag)
    joint_feats_ = collect_features(dataset, models_config, dataset_flag)
    joint_feats_ = torch.tensor(joint_feats_, dtype=torch.float)
    if labels is not None:
        label_ = torch.tensor(labels, dtype=torch.long)
        dataset = TensorDataset(joint_feats_, label_)
    else:
        dataset = TensorDataset(joint_feats_)
    return dataset, joint_feats_.shape[-1]


def collect_features(
    dataset: Dataset, models_config: list[dict], dataset_flag: str
) -> list[list]:

    logging.info("Collecting models outputs on dataset %s ...", dataset_flag)

    feats_cache_path = generate_cache_path(
        data=dataset, filename=f"joint_feats_{dataset_flag}.tmp", model=models_config
    )
    if not os.path.exists(feats_cache_path):

        joint_feats = [[] for _ in range(len(dataset))]
        for model_config in tqdm(models_config, desc="Overall progress"):
            if model_config["for_joint_extreme_short"] is False:
                logging.info("Skip model: %s", model_config["id"])
                continue

            logging.info(
                "Extracting feature using model: %s(%s) %s",
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
            )  # type: dict[list]

            assert isinstance(results, dict), "Invalid type for inference results"
            for key, value in results.items():
                if key != "preds":
                    assert len(dataset) == len(
                        value
                    ), "Unexpected size for inference results"
                    if isinstance(value[0], list) and isinstance(
                        value[0][0], (int, float)
                    ):
                        for i, feat in enumerate(joint_feats):
                            feat += value[i]
                    elif isinstance(value[0], (int, float)):
                        for i, feat in enumerate(joint_feats):
                            feat.append(value[i])
                    else:
                        print("Invalid type for feature elements")

        with open(feats_cache_path, "wb") as f_cache:
            pickle.dump(joint_feats, f_cache)
            logging.info("Save joint features to cache `%s`", feats_cache_path)
        # feats_log_path = os.path.join(Config.LOG_DIR, f"joint_feats_{dataset_flag}.txt")
        # with open(feats_log_path, "w", encoding="utf-8") as f:
        #     json.dump(joint_feats, f, ensure_ascii=False, indent=4)
        #     logging.info("Save joint features (readable) to file `%s`", feats_log_path)

    else:
        logging.info("Reading joint features from cache `%s`", feats_cache_path)
        with open(feats_cache_path, "rb") as f_cache:
            joint_feats = pickle.load(f_cache)

    return joint_feats


class MLPForAIGTDetection(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(MLPForAIGTDetection, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class JointFeaturesMLP(BaseNeuralModel):

    desc = repr("MLP (joint features) for AIGT detection")

    def __init__(
        self,
        model_path: str,
        models_config: list[dict],
        feat_dim: int,
        hidden_dim: int = 512,
    ):
        self.model_path = model_path
        self.model = MLPForAIGTDetection(
            input_dim=feat_dim, hidden_dim=hidden_dim, output_dim=2
        )
        self.model.load_state_dict(torch.load(model_path, weights_only=True))
        self.models_config = models_config

    def __call__(self, text: str, **kwargs):
        return self.inference_example(text, kwargs)

    def inference_example(self, text: str, device="cpu", **kwargs):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        self.model.eval()
        pass

    def inference_dataset(
        self, dataset: Dataset, dataset_flag: str, device="cpu", **kwargs
    ):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        preds_cache_path = generate_cache_path(
            data=dataset,
            filename=f"{self.__class__.__name__}.tmp",
            model_name=self.model_path,
        )

        prepared_dataset, _ = prepare_data(
            dataset=dataset,
            models_config=self.models_config,
            dataset_flag=dataset_flag,
        )
        dev_loader = DataLoader(prepared_dataset, batch_size=16)

        if not os.path.exists(preds_cache_path):
            logging.info("Using device: %s", device)
            self.model.to(device)
            self.model.eval()
            preds = []

            with torch.no_grad():
                for (x_batch,) in dev_loader:
                    x_batch = x_batch.to(device)
                    logits = self.model(x_batch)
                    pred = torch.argmax(logits, dim=-1).cpu()
                    preds.extend(pred.tolist())

            cache_data = {"preds": preds}
            with open(preds_cache_path, "wb") as f_cache:
                pickle.dump(cache_data, f_cache)
                logging.info(
                    "Save model prediction results to cache `%s`", preds_cache_path
                )

        else:
            logging.info("Reading prediction results from cache `%s`", preds_cache_path)
            with open(preds_cache_path, "rb") as f_cache:
                cache_data = pickle.load(f_cache)

        return cache_data


if __name__ == "__main__":

    trainset_path = os.path.join(Config.RAW_DATA["train"])
    devset_path = os.path.join(Config.RAW_DATA["dev"])
    logging.info("Loading train-set from %s ...", repr(trainset_path))
    logging.info("Loading dev-set from %s ...", repr(devset_path))
    dataset_dict = load_dataset(
        "json", data_files={"train": trainset_path, "dev": devset_path}
    )
    print(dataset_dict)

    extra_kwargs = {
        "labels": dataset_dict["dev"]["label"],
        "refer_dataset": dataset_dict["train"],
    }
    models_config_ = Config.prepare_models_config(**extra_kwargs)
    _, dev_feat_dim_ = prepare_data(
        dataset=dataset_dict["dev"],
        models_config=models_config_,
        dataset_flag="dev",
    )

    mlp_model_path = os.path.join(
        Config.CKPT_DIR, "joint-features-mlp_04-07_14-48_957.pth"
    )
    mlp = JointFeaturesMLP(
        model_path=mlp_model_path,
        models_config=models_config_,
        feat_dim=dev_feat_dim_,
        hidden_dim=512,
    )

    results_ = mlp.inference_dataset(
        dataset=dataset_dict["dev"], dataset_flag="dev", device="cuda"
    )

    evaluator = PerformanceEvaluator()
    labels_ = dataset_dict["dev"]["label"]
    evaluator.calculate_classification_performance(labels_, results_["preds"])
