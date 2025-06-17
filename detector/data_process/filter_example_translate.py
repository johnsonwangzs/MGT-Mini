import os.path
import logging
import json
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from detector.config import Config

print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("Number of visible GPUs:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f"CUDA logical device {i} -> {torch.cuda.get_device_name(i)}")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def llm_check_translate(gen_kwargs: dict, dataset: list, check_path: str):
    """few-shot提示的回译使用mbart-large-50-many-to-many-mmt模型对原文进行中->英->中翻译得到"""
    for example in tqdm(dataset, desc="Inferring ..."):
        prompt = """回译是将中文文本翻译成英文再翻译回中文的过程。由回译产生的中文文本通常会有一些不同于人工书写的痕迹，如句式生硬、用词不自然或语法结构不符合本地化表达习惯，有时还残留有英文单词和多余的空格。
        
        下面是几个例子，展示了原始文本与回译后的文本：

        示例1
        原文：每天清晨，它都会飞到森林里最高的树枝上，用清脆悦耳的歌声唤醒沉睡的森林。它的歌声让其他鸟儿羡慕不已，甚至连风儿都停下来倾听。
        回译：每天早晨，它飞到森林的最高树枝上，唤醒睡着的森林，用一种鲜艳、愉快的歌声，使其他鸟类羡慕，甚至风也停下来倾听。 

        示例2
        原文：孔雀正站在一棵大树下，展开它那五彩斑斓的尾羽，阳光透过树叶洒在羽毛上，闪烁着耀眼的光芒。它昂首挺胸，骄傲地环顾四周，仿佛在向整个世界展示自己的美丽。
        回译：孔雀站在一棵大树的下面，伸展着五色尾巴，太阳在羽毛的叶子里闪烁着，闪烁着灿烂的光芒，他骄傲地环顾四周，仿佛要向全世界展示他的美丽。

        示例3
        原文：小明和小红是村里最要好的朋友，他们总是形影不离。那天，阳光透过树叶洒在地上，斑驳的光影随着微风轻轻摇曳，仿佛在为他们伴舞。
        回译：小明和小红是村子里最好的朋友，他们总是看不见的。 那天，太阳在树叶上闪耀着，闪烁的灯光和风轻轻地摇动着，仿佛是为他们跳舞的。

        示例4
        原文：青蛙先生很爱干净，他每天要用很多水洗菜、洗衣服、拖地板。他的家总是闪闪发光，一尘不染。邻居们常常夸赞他的勤劳和整洁，青蛙先生也因此感到非常自豪。
        回译：弗格先生喜欢清洁。他每天用大量的水洗碗、衣服和地板。他的房子总是光亮干净。 邻居们经常赞扬他的勤奋工作和清洁,弗格先生对此非常自豪。

        示例5
        原文：赵奢是战国时期赵国的一代名将，以智勇双全、治军严明著称。他不仅善于用兵，更以公正无私、体恤百姓而闻名。然而，赵奢的成名之路并非一帆风顺，他的故事中蕴含着深刻的道德启示。
        回译：Zhao She是战国时期赵朝的伟大将军，他不仅以兵力使用能力而又以不偏不倚和自私而闻名，然而Zhao She的名声之路并不平坦，他的故事蕴含着深刻的道德启示。 

        以下是一段中文文本，请判断它是否是由“回译”操作生成的文本。

        文本：
        {text}

        请只回答“【是】”或“【否】”。
        """
        query = prompt.format(text=example["text"])
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": query}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            comment = tokenizer.decode(outputs[0], skip_special_tokens=True)

            if "【是】" in comment:
                judge = 1
            elif "【否】" in comment:
                judge = 0
            else:
                judge = -1

            json_obj = {
                "id": example["id"],
                "judge": judge,
                "text": example["text"],
                "comment": comment,
            }
            with open(check_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":

    device = "cuda"
    model_path = os.path.join(Config.CKPT_DIR, "Qwen2.5-32B-Instruct")
    dataset_path = Config.RAW_DATA["test"]
    check_path = os.path.join(Config.DATA_DIR, "test_assist_translate.jsonl")

    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = (
        AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        .to(device)
        .eval()
    )
    gen_kwargs = {"max_length": 10000, "do_sample": False}

    llm_check_translate(gen_kwargs, dataset, check_path)
