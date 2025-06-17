# Copyright (c) 2025 ASCII Lab (CAS-IIE). All rights reserved.
# This code is submitted as part of [NLPCC25-Task1].
# Use of this code is permitted only for evaluation purposes related to the competition.
import os.path
import logging
import pickle
from typing import override
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizer,
)
from peft import PeftModel
from datasets import load_dataset, Dataset
from tqdm import tqdm
from detector.config import Config
from detector.utils import generate_cache_path
from detector.model.neural_based.base_neural_model import BaseNeuralModel
from detector.eval.evaluator import PerformanceEvaluator


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class LLMForAIGTDetection(BaseNeuralModel):

    desc = repr("Supervise fine-tuned LLM for AIGT detection")

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        model_cls: PreTrainedModel,
        tokenizer_cls: PreTrainedTokenizer,
    ):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.model_cls = model_cls
        self.tokenizer_cls = tokenizer_cls
        self.lora_sft_path = None
        self.default_prompt = "请判断以下文本是由人类撰写还是由机器生成："
        self.default_system_prompt = (
            "你是一个善于区分机器生成文本和人类撰写文本的专家。"
        )
        self.gen_kwargs = {
            "max_new_tokens": 32,  # 控制输出长度，不宜过长
            "do_sample": False,  # 不采样，使用贪心或 beam search 更确定
        }

    def _load_model(self):
        # pylint: disable=attribute-defined-outside-init
        logging.info("Loading LLM: %s", self.model_path)
        self.model = self.model_cls.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"

    def inference_example(
        self,
        text: str,
        add_default_prompt: bool = True,
        add_default_system_prompt: bool = False,
        ignore_double_newline: bool = True,
        device="cpu",
        **kwargs,
    ):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        self._load_model()
        self.model.to(device)

        inputs = self.tokenizer.apply_chat_template(
            self._generate_template(
                text,
                add_default_prompt=add_default_prompt,
                add_default_system_prompt=add_default_system_prompt,
                ignore_double_newline=ignore_double_newline,
            ),
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **self.gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = self._extract_answer_from_response(response)

        print(f"Model response: {response}")
        print(f"Prediction: {answer}")

    def inference_dataset(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        add_default_prompt: bool = False,
        add_default_system_prompt: bool = True,
        ignore_double_newline: bool = True,
        device="cpu",
        **kwargs,
    ):
        logging.info("Performing inference using %s ...", self.__class__.__name__)
        preds_cache_path = generate_cache_path(
            data=dataset,
            filename=f"{self.__class__.__name__}.tmp",
            model_name=self.model_path,
            lora_path=self.lora_sft_path,
            add_default_prompt=add_default_prompt,
            add_default_system_prompt=add_default_system_prompt,
            ignore_double_newline=ignore_double_newline,
        )

        if not os.path.exists(preds_cache_path):
            logging.info("Using device: %s", device)
            self._load_model()
            self.model.to(device)
            preds = []

            if batch_size == 1:
                for example in tqdm(dataset["text"], desc="Inferring ..."):
                    inputs = self.tokenizer.apply_chat_template(
                        self._generate_template(
                            example,
                            add_default_prompt=add_default_prompt,
                            add_default_system_prompt=add_default_system_prompt,
                            ignore_double_newline=ignore_double_newline,
                        ),
                        add_generation_prompt=True,
                        tokenize=True,
                        return_tensors="pt",
                        return_dict=True,
                    ).to(device)

                    with torch.no_grad():
                        outputs = self.model.generate(**inputs, **self.gen_kwargs)
                        outputs = outputs[:, inputs["input_ids"].shape[1] :]
                        pred = self.tokenizer.decode(
                            outputs[0], skip_special_tokens=True
                        )
                        preds.append(self._extract_answer_from_response(pred))

            else:
                assert (
                    len(dataset) % batch_size == 0
                ), f"Dataset length is not divisible by {batch_size=}"

                batch_templates = []
                for example in tqdm(dataset["text"], desc="Inferring ..."):
                    batch_templates.append(
                        self._generate_template(
                            example,
                            add_default_prompt=add_default_prompt,
                            add_default_system_prompt=add_default_system_prompt,
                            ignore_double_newline=ignore_double_newline,
                        )
                    )

                    if len(batch_templates) == batch_size:
                        inputs = self.tokenizer.apply_chat_template(
                            batch_templates,
                            add_generation_prompt=True,
                            tokenize=True,
                            padding=True,
                            truncation=True,
                            return_tensors="pt",
                            return_dict=True,
                        ).to(device)

                        with torch.no_grad():
                            outputs = self.model.generate(**inputs, **self.gen_kwargs)
                            outputs = outputs[:, inputs["input_ids"].shape[1] :]
                            for i in range(batch_size):
                                pred = self.tokenizer.decode(
                                    outputs[i], skip_special_tokens=True
                                )
                                preds.append(self._extract_answer_from_response(pred))

                        batch_templates = []

            cache_data = {"preds": preds}
            with open(preds_cache_path, "wb") as f_cache:
                pickle.dump(cache_data, f_cache)
                logging.info(
                    "Save model prediction results to cache `%s`", preds_cache_path
                )
        else:
            # 直接读取缓存推理结果
            logging.info("Reading prediction results from cache `%s`", preds_cache_path)
            with open(preds_cache_path, "rb") as f_cache:
                cache_data = pickle.load(f_cache)

        return cache_data

    @staticmethod
    def _extract_answer_from_response(text: str) -> int:

        return 1 if "机器" in text else 0

    def _generate_template(
        self,
        text: str,
        add_default_prompt: bool,
        add_default_system_prompt: bool,
        ignore_double_newline: bool,
    ):
        """构造问答模板"""
        if ignore_double_newline is True:
            text = text.replace("\n\n", "")
        prompt_text = (
            self.default_prompt + "\n" + text if add_default_prompt is True else text
        )
        template = []
        if add_default_system_prompt is True:
            template.append({"role": "system", "content": self.default_system_prompt})
        template.append({"role": "user", "content": prompt_text})
        # print(template)
        return template


class GLM4ForAIGTDetection(LLMForAIGTDetection):

    desc = repr("Lora-sft glm-4 model for AIGT detection")

    def __init__(
        self,
        base_model_path: str,
        lora_sft_path: str,
        base_tokenizer_path: str,
        base_model_cls: PreTrainedModel,
        base_tokenizer_cls: PreTrainedTokenizer,
    ):
        super().__init__(
            base_model_path, base_tokenizer_path, base_model_cls, base_tokenizer_cls
        )
        self.lora_sft_path = lora_sft_path
        self.default_system_prompt = (
            "你是一个善于区分机器生成文本和人类撰写文本的专家。"
        )
        self.default_prompt = "请判断以下文本是由人类撰写还是由机器生成："
        self.gen_kwargs["top_p"] = None
        self.gen_kwargs["temperature"] = None

    @override
    def _load_model(self):
        # pylint: disable=attribute-defined-outside-init
        logging.info("Loading LLM: %s", self.model_path)
        self.model = self.model_cls.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logging.info("Loading lora-sft weights: %s", self.lora_sft_path)
        self.model = PeftModel.from_pretrained(self.model, self.lora_sft_path)
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"

    @staticmethod
    @override
    def _extract_answer_from_response(text: str) -> int:
        """从模型的回答（文本）中提取答案并转换为预测值（0/1）"""
        with open("test_predict.txt", "a", encoding="utf-8") as f:
            f.write(repr(text))
            f.write("\n")
        return 1 if "机器" in text else 0


class Qwen25ForAIGTDetection(LLMForAIGTDetection):

    desc = repr("Lora-sft Qwen2.5-7B-Instruct model for AIGT detection")

    def __init__(
        self,
        base_model_path: str,
        lora_sft_path: str,
        base_tokenizer_path: str,
        base_model_cls: PreTrainedModel,
        base_tokenizer_cls: PreTrainedTokenizer,
    ):
        super().__init__(
            base_model_path, base_tokenizer_path, base_model_cls, base_tokenizer_cls
        )
        self.lora_sft_path = lora_sft_path
        self.default_system_prompt = (
            "你是一个善于区分机器生成文本和人类撰写文本的专家。"
        )
        self.default_prompt = "请判断以下文本是由人类撰写还是由机器生成："
        self.gen_kwargs["top_p"] = None
        self.gen_kwargs["top_k"] = None
        self.gen_kwargs["temperature"] = None

    @override
    def _load_model(self):
        # pylint: disable=attribute-defined-outside-init
        logging.info("Loading LLM: %s", self.model_path)
        self.model = self.model_cls.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logging.info("Loading lora-sft weights: %s", self.lora_sft_path)
        self.model = PeftModel.from_pretrained(self.model, self.lora_sft_path)
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"

    @staticmethod
    @override
    def _extract_answer_from_response(text: str) -> int:
        """从模型的回答（文本）中提取答案并转换为预测值（0/1）"""
        return 1 if "机器" in text else 0


class Llama3ChineseForAIGTDetection(LLMForAIGTDetection):

    desc = repr("Lora-sft Llama3.1-8B-Chinese-Chat model for AIGT detection")

    def __init__(
        self,
        base_model_path: str,
        lora_sft_path: str,
        base_tokenizer_path: str,
        base_model_cls: PreTrainedModel,
        base_tokenizer_cls: PreTrainedTokenizer,
    ):
        super().__init__(
            base_model_path, base_tokenizer_path, base_model_cls, base_tokenizer_cls
        )
        self.lora_sft_path = lora_sft_path
        self.default_system_prompt = (
            "你是一个善于区分机器生成文本和人类撰写文本的专家。"
        )
        self.default_prompt = "请判断以下文本是由人类撰写还是由机器生成："
        self.gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id

    @override
    def _load_model(self):
        # pylint: disable=attribute-defined-outside-init
        logging.info("Loading LLM: %s", self.model_path)
        self.model = self.model_cls.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logging.info("Loading lora-sft weights: %s", self.lora_sft_path)
        self.model = PeftModel.from_pretrained(self.model, self.lora_sft_path)
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"

    @staticmethod
    @override
    def _extract_answer_from_response(text: str) -> int:
        """从模型的回答（文本）中提取答案并转换为预测值（0/1）"""
        return 1 if "机器" in text else 0


class Llama3ForAIGTDetection(LLMForAIGTDetection):

    desc = repr("Lora-sft Llama3.1-8B-Instruct model for AIGT detection")

    def __init__(
        self,
        base_model_path: str,
        lora_sft_path: str,
        base_tokenizer_path: str,
        base_model_cls: PreTrainedModel,
        base_tokenizer_cls: PreTrainedTokenizer,
    ):
        super().__init__(
            base_model_path, base_tokenizer_path, base_model_cls, base_tokenizer_cls
        )
        self.lora_sft_path = lora_sft_path
        self.default_system_prompt = (
            "你是一个善于区分机器生成文本和人类撰写文本的专家。"
        )
        self.default_prompt = "请判断以下文本是由人类撰写还是由机器生成："
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.gen_kwargs["pad_token_id"] = self.tokenizer.eos_token_id
        self.gen_kwargs["top_p"] = None
        self.gen_kwargs["temperature"] = None

    @override
    def _load_model(self):
        # pylint: disable=attribute-defined-outside-init
        logging.info("Loading LLM: %s", self.model_path)
        self.model = self.model_cls.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        logging.info("Loading lora-sft weights: %s", self.lora_sft_path)
        self.model = PeftModel.from_pretrained(self.model, self.lora_sft_path)
        self.tokenizer = self.tokenizer_cls.from_pretrained(
            self.tokenizer_path, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"

    @staticmethod
    @override
    def _extract_answer_from_response(text: str) -> int:
        """从模型的回答（文本）中提取答案并转换为预测值（0/1）"""
        return 1 if "机器" in text else 0


if __name__ == "__main__":

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # glm-4-9b-chat
    glm_path = os.path.join(Config.CKPT_DIR, "glm-4-9b-chat")
    glm_lora_path = os.path.join(Config.CKPT_DIR, "glm-4-9b-chat-lora/checkpoint-2000")
    model = GLM4ForAIGTDetection(
        base_model_path=glm_path,
        lora_sft_path=glm_lora_path,
        base_tokenizer_path=glm_path,
        base_model_cls=AutoModelForCausalLM,
        base_tokenizer_cls=AutoTokenizer,
    )

    # Qwen2.5-7B-Instruct
    # qwen_path = os.path.join(Config.CKPT_DIR, "Qwen2.5-7B-Instruct")
    # qwen_lora_path = os.path.join(Config.CKPT_DIR, "Qwen2.5-7B-Instruct-lora")
    # model = Qwen25ForAIGTDetection(
    #     base_model_path=qwen_path,
    #     lora_sft_path=qwen_lora_path,
    #     base_tokenizer_path=qwen_path,
    #     base_model_cls=AutoModelForCausalLM,
    #     base_tokenizer_cls=AutoTokenizer,
    # )

    # Llama-3.1-8B-Chinese-Chat
    # llama_ch_path = os.path.join(Config.CKPT_DIR, "Llama-3.1-8B-Chinese-Chat")
    # llama_ch_lora_path = os.path.join(Config.CKPT_DIR, "Llama3.1-8B-Chinese-Chat-lora")
    # model = Llama3ChineseForAIGTDetection(
    #     base_model_path=llama_ch_path,
    #     lora_sft_path=llama_ch_lora_path,
    #     base_tokenizer_path=llama_ch_path,
    #     base_model_cls=AutoModelForCausalLM,
    #     base_tokenizer_cls=AutoTokenizer,
    # )

    # Llama-3.1-8B-Instruct
    # llama_path = os.path.join(Config.CKPT_DIR, "Llama-3.1-8B-Instruct")
    # llama_lora_path = os.path.join(Config.CKPT_DIR, "Llama-3.1-8B-Instruct-lora")
    # model = Llama3ForAIGTDetection(
    #     base_model_path=llama_path,
    #     lora_sft_path=llama_lora_path,
    #     base_tokenizer_path=llama_path,
    #     base_model_cls=AutoModelForCausalLM,
    #     base_tokenizer_cls=AutoTokenizer,
    # )

    # 机器测试样本
    test_text = "同学是靖江的，推荐了我们吃这家羊肉，店面很大，大堂明亮，关键是不会有很重羊膻味儿，这一点真的让我很惊喜。很多地方的羊肉店一进门就能闻到浓重的羊膻味，而这家店则完全没有这种困扰。羊肉煮得非常软烂，入口即化，调味也恰到好处，不会掩盖住羊肉自身的鲜美。我们点了几道招牌菜，包括羊肉火锅、红烧羊肉和孜然羊排，味道都非常好。火锅的羊肉切片均匀，煮熟后口感极佳，配上清新的蔬菜和独特的蘸酱，真是让人食欲大开。红烧羊肉色泽诱人，汤汁浓稠，非常入味，和米饭是绝配。孜然羊排外焦里嫩，香气扑鼻，配上一杯清凉饮料，实在是完美的享受。服务方面，服务员都很热情，点菜时会耐心介绍每一道菜的特色，还贴心地帮我们推荐了几道适合夏季的清爽小菜。总体来说，这次的用餐体验非常愉快，既满足了口腹之欲，也感受到了靖江当地人的热情好客。如果以后有机会，一定还会再来光顾，也会带上更多朋友一起来分享这份美味。推荐给所有喜欢羊肉的朋友们，这真的是一家不可错过的餐厅！"

    # 人类测试样本
    # test_text = "同学是靖江的，推荐了我们吃这家羊肉，店面很大，大堂明亮，关键是不会有很重羊膻味儿…能看到后厨，大师傅们忙的热火朝天，后厨蛮干净的，印象分不错…总共三层楼，很大，包间很多，每一层有专门的服务员，点餐时有专业人员推荐，帮你搭配，很不错…羊汤，一大盆，半只羊排，很扎实啊，一块块肉都很大，而且嫩，羊汤很浓香…手拿骨头肯肉，怎一个爽字了得！！大蒜羊杂，符合苏南人口味，甜甜的，羊杂没有异味，口感特别好！！！只是大蒜没经过霜冻，有一点点老…砂锅羊血，赞一个，我的小伙伴一个人吃了半锅，脆嫩的口感，还有点像布丁，没有羊血的异味，调味适中，真的很好吃！色拉解腻，就是一盘草～也蛮好吃的还有他红烧的白丝鱼！很新鲜，入口即话，比我吃过清蒸的别有风味，里面的毛豆子很酥烂，淡淡的甜味…菠菜时新菜，都小小嫩嫩的，不会有涩的口号，补充维生素！！！最后，他家是烤全羊料理，还有很多其他的菜，推荐给大家………"
    # test_text = "【地址】世贸广场7楼，钟表后面的电梯可以直达。交通便利，地理位置好找。【环境】这家秀玉算是比较大的一家，可以选择自己喜欢的座位，环境不错，很适合聚会聊天。【服务】服务很热情，从一进门开始就有人主动询问，一路有人把你带到座位上。整个用餐过程的服务都做到及时，主动，热情，挺好的。【口味】连锁餐厅，口味有一定保障。虽然没有什么创新菜式，但是用餐，聚会，下午茶什么的没问题。老公喜欢这里的牛排，因为柔嫩分量足，吃得很饱；我喜欢这里的煲仔饭和泰皇炒饭，还有水果沙拉，吐司。吃完饭如果不想接着逛街，可以休息会，2点就有下午茶了，点上一壶茶，边喝边聊。唯一缺点就是世贸禁止明火，茶下面不能有蜡烛保温，这个天气，茶一会就冷了。但是我觉得可以用个插电的保温垫，服务是需要更细致，用心的！"

    # model.inference_example(
    #     test_text,
    #     add_default_prompt=True,
    #     add_default_system_prompt=True,
    #     ignore_double_newline=True,
    #     device=device,
    # )

    logging.info("Loading dataset ...")
    trainset_path = os.path.join(Config.RAW_DATA["train"])
    devset_path = os.path.join(Config.RAW_DATA["dev"])
    logging.info("Loading train-set from %s ...", repr(trainset_path))
    logging.info("Loading dev-set from %s ...", repr(devset_path))
    dataset_dict = load_dataset(
        "json", data_files={"train": trainset_path, "dev": devset_path}
    )
    print(dataset_dict)
    devset = dataset_dict["dev"]

    results = model.inference_dataset(
        devset,
        batch_size=1,
        add_default_prompt=True,
        add_default_system_prompt=False,
        ignore_double_newline=True,
        device=device,
    )
    evaluator = PerformanceEvaluator()
    labels = dataset_dict["dev"]["label"]
    evaluator.calculate_classification_performance(labels, results["preds"])

    for i, pred in enumerate(results["preds"]):
        if pred != labels[i]:
            print(f"{i} ", end="")

    # 测试直接使用llamafactory进行的预测
    # evaluator = PerformanceEvaluator()
    # evaluator.evaluate_from_predict_file(
    #     os.path.join(
    #         Config.LOG_DIR,
    #         "predict_result/glm4_lora_sft_3_2000/generated_predictions.jsonl",
    #     )
    # )
