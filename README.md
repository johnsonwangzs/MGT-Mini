# mgtd-sys

项目代码仓库（将在提交截止时公开）：[https://github.com/johnsonwangzs/mgtd-sys](https://github.com/johnsonwangzs/mgtd-sys)

方案设计说明（将在提交截止时公开）：[https://www.yuque.com/johnsonwangzs/ikuc6g/yufoovloguego23t](https://www.yuque.com/johnsonwangzs/ikuc6g/yufoovloguego23t)

## 使用方法
### 必要准备
#### 环境要求
见 `requirements.txt`。建议基于 `environment.yaml` 安装新的 Conda 环境。

#### 开源模型权重下载
下载以下开源模型权重，存放到 `detector/ckpt/`目录下：

1. chinese-roberta-wwm-ext-large 模型：[https://huggingface.co/hfl/chinese-roberta-wwm-ext-large](https://huggingface.co/hfl/chinese-roberta-wwm-ext-large)
2. chinese-bert-wwm-ext 模型：[https://huggingface.co/hfl/chinese-bert-wwm-ext](https://huggingface.co/hfl/chinese-bert-wwm-ext)
3. glm-4-9b-chat 模型：[https://huggingface.co/THUDM/glm-4-9b-chat](https://huggingface.co/THUDM/glm-4-9b-chat)
4. Qwen2.5-7B 模型：[https://huggingface.co/Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)
5. Qwen2.5-7B-Instruct 模型：[https://huggingface.co/Qwen/Qwen2.5-7B](https://huggingface.co/Qwen/Qwen2.5-7B)

#### 训练权重下载
下载以下训练权重，存放到 `detector/ckpt/` 目录下：

1. chinese-roberta 的 sft 微调：[https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/chinese-roberta-sft_03-21_16-39_938](https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/chinese-roberta-sft_03-21_16-39_938)
2. chinese-bert 的 sft 微调：[https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/chinese-bert-sft_03-28_20-17_908](https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/chinese-bert-sft_03-28_20-17_908)
3. glm-4-9b-chat 的 lora 微调：[https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/glm-4-9b-chat-lora](https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/glm-4-9b-chat-lora)
4. Qwen2.5-7B-Instruct 的 lora 微调：[https://huggingface.co/ZhaoCamera/mgtd-sys-model-weights/tree/main/Qwen2.5-7B-Instruct-lora-2](https://huggingface.co/ZhaoCamera/mgtd-sys-model-weights/tree/main/Qwen2.5-7B-Instruct-lora-2)
5. Qwen2.5-7B-Instruct 的 lora 微调（超短文本）：[https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/Qwen2.5-7B-Instruct-lora-extreme_short](https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/Qwen2.5-7B-Instruct-lora-extreme_short)
6. Qwen2.5-7B-Instruct 的 lora 微调（短文本）：[https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/Qwen2.5-7B-Instruct-lora-short](https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/tree/main/Qwen2.5-7B-Instruct-lora-short)
5. 混合特征模型：[https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/blob/main/joint-features-mlp_04-07_14-48_957.pth](https://huggingface.co/johnsonwangzs/mgtd-sys-model-weights/blob/main/joint-features-mlp_04-07_14-48_957.pth)

### 运行程序
注意：

1. 必须在项目根目录使用`python -m`运行
2. 初次运行时，必须指定两张 V100 级别以上的 GPU

```bash
cd mgtd-sys
CUDA_VISIBLE_DEVICES=0,1 python -m detector.model.ensemble_model
```

## 文件说明
### model
`base_model.py`：抽象类 `BaseModel`，该类拥有两个抽象方法：数据集推理 `inference_dataset`和单样本推理 `inference_example`

`ensemble_model.py`：集成模型，实现全流程控制。定义了各子模型的配置参数并实例化各子模型，实现各子模型的串行推理，进行投票得出最终预测结果

`rule_based/`：定义基于规则的模型

+ `base_rule_model.py`：定义了 `BaseRuleModel`，是所有规则模型的父类，继承自 `BaseModel`
+ `common_phrase.py`：对于待测文本，检测是否存在常见的机器短语
+ `common_token.py`：对于待测文本，比较其 tokens 在人类偏好词频表和机器偏好词频表中的出现数量
+ `consecutive_punctuation.py`：对于待测文本，检测其中是否存在连续的标点符号
+ `special_token.py`：对于待测文本，检测其中是否存在特殊字符，例如连续两个换行符的情况
+ `sentence_segment.py`：对于待测文本，分析连续子句（逗号间隔视为子句，句号视为结束）的数量分布

`neural_based/`：定义基于深度神经网络的模型

+ `base_neural_model.py`：定义了 `BaseNeuralModel`，是所有神经模型的父类，继承自 `BaseModel`
+ `llm_detector.py`：定义了 `LLMForAIGTDetection`类，是所有基于 LLM 微调的模型的父类
+ `chinese_bert_detector.py`：定义了 `ChineseBertForAIGTDetection`类，是基于 Bert 的检测器
+ `fast_detectgpt.py`：定义了 `FastDetectGPT`类，实现了 Fast-DetectGPT 零样本检测方法（[https://arxiv.org/abs/2310.05130](https://arxiv.org/abs/2310.05130)）
+ `binoculars.py`：定义了 `BinocularsDetector`类，实现了 Binoculars 零样本检测方法（[https://arxiv.org/abs/2401.12070](https://arxiv.org/abs/2401.12070)）
+ `joint_features.py`：定义了 `JointFeaturesMLP`类，一种混合特征 MLP 模型
+ `train/`：训练脚本
    - `chinese_bert_sft.py`：`ChineseBertForAIGTDetection`模型的微调代码
    - `mlp_train.py`：`JointFeaturesMLP`模型的微调代码
    - `lora-sft`：使用 LLaMA-Factory 进行 LLM lora 微调的配置

### data_process
+ `build_sft_data.py`：基于已有训练集，构建 lora 微调格式的数据
+ `build_short_text_dataset.py`：基于已有训练集，构建短文本数据
+ `build_vote_rules.py`：构建投票规则
+ `filter_example_translate.py`：利用few-shot提示大模型找出潜在的回译样本

### 其他目录
`eval/evaluator.py`：定义了评估器 `PerformanceEvaluator`

`config.py`：项目全局常量配置（项目目录和各个模型的配置）

`utils.py`：工具函数





