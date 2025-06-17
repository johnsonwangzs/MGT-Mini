# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """基本模型类"""

    @abstractmethod
    def inference_example(self, text: str, **kwargs):
        """单样本推理"""

    @abstractmethod
    def inference_dataset(self, dataset, **kwargs) -> dict[str, list]:
        """在数据集上推理

        Args:
            dataset (_type_): _description_

        Returns:
            dict[str, list]: 返回值为一个字典。
            - 每个字段为一个列表，长度与数据集大小一致
            - 至少包含一个字段`preds`表示模型在整个数据集上的预测结果（0或1）
            - 可以包含若干其余的字段，每个字段表示一种特征
        """
