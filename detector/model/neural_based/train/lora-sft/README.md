本目录中保存了使用LLaMA-Factory工具进行lora微调的相关配置和数据文件，以便于模型的复现。

- `glm4_lora_sft.yaml`：对glm-4-9b-chat模型进行lora微调的yaml配置文件，使用时复制到`LLaMA-Factory/examples/train_lora`目录下
- `glm4-9b-chat.yaml`：使用glm-4-9b-chat模型进行推理的yaml配置文件，使用时复制到`LLaMA-Factory/examples/inference`目录下
- `nlpcc25_task1_train.json`：转换为Alpaca格式后的nlpcc25-task1训练集（train.json），见`mgtd-sys/data/data_sft`目录，使用时复制到`LLaMA-Factory/data`目录下