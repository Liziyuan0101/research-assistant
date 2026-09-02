"""
LoRA Trainer
Qwen2.5-7B LoRA 微调训练器

支持：
- PEFT LoRA 微调
- 学术写作任务训练
- 训练过程监控和保存
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime

import torch

try:
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        BitsAndBytesConfig
    )
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("⚠️ transformers not installed")

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        prepare_model_for_kbit_training,
        TaskType
    )
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠️ peft not installed")

try:
    from datasets import Dataset, load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    Dataset = None
    load_dataset = None

try:
    from trl import SFTTrainer
    HAS_TRL = True
except ImportError:
    HAS_TRL = False
    print("⚠️ trl not installed")


@dataclass
class TrainingConfig:
    """训练配置"""
    # 模型配置
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    tokenizer_name: Optional[str] = None
    
    # LoRA 配置
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    
    # 量化配置
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False
    
    # 训练配置
    output_dir: str = "output/lora_model"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    
    # 序列配置
    max_seq_length: int = 2048
    
    # 保存配置
    save_steps: int = 100
    eval_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3
    
    # 其他
    seed: int = 42
    fp16: bool = True
    bf16: bool = False
    gradient_checkpointing: bool = True
    
    def to_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'lora_r': self.lora_r,
            'lora_alpha': self.lora_alpha,
            'lora_dropout': self.lora_dropout,
            'target_modules': self.target_modules,
            'use_4bit': self.use_4bit,
            'num_train_epochs': self.num_train_epochs,
            'learning_rate': self.learning_rate,
            'max_seq_length': self.max_seq_length,
            'per_device_train_batch_size': self.per_device_train_batch_size,
            'gradient_accumulation_steps': self.gradient_accumulation_steps
        }


class LoRATrainer:
    """
    LoRA 微调训练器
    
    针对 Qwen2.5-7B 进行学术写作任务的 LoRA 微调
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
        # 确保输出目录存在
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    def setup_model(self):
        """设置模型和分词器"""
        if not HAS_TRANSFORMERS or not HAS_PEFT:
            raise ImportError("transformers and peft are required")
        
        print(f"🔄 Loading model: {self.config.model_name}")
        
        # 量化配置
        if self.config.use_4bit:
            compute_dtype = getattr(torch, self.config.bnb_4bit_compute_dtype)
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=self.config.use_nested_quant
            )
        else:
            bnb_config = None
        
        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16 if self.config.fp16 else torch.float32
        )
        
        # 准备模型进行 k-bit 训练
        if self.config.use_4bit:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # 加载分词器
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 配置 LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        # 应用 LoRA
        self.model = get_peft_model(self.model, lora_config)
        
        # 打印可训练参数
        self.model.print_trainable_parameters()
        
        print("✅ Model setup complete")
    
    def prepare_dataset(self, data_path: str) -> Dataset:
        """准备训练数据集"""
        if not HAS_DATASETS:
            raise ImportError("datasets library required")
        
        print(f"📂 Loading dataset from: {data_path}")
        
        # 加载数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 处理数据格式
        processed_data = []
        
        for item in data:
            if 'messages' in item:
                # Chat 格式
                text = self._format_chat_messages(item['messages'])
            else:
                # Instruction 格式
                text = self._format_instruction(item)
            
            processed_data.append({'text': text})
        
        dataset = Dataset.from_list(processed_data)
        
        print(f"✅ Loaded {len(dataset)} training samples")
        return dataset
    
    def _format_chat_messages(self, messages: List[Dict]) -> str:
        """格式化聊天消息为训练文本"""
        # Qwen 格式
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == 'user':
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == 'assistant':
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        return formatted
    
    def _format_instruction(self, item: Dict) -> str:
        """格式化指令数据为训练文本"""
        instruction = item.get('instruction', '')
        input_text = item.get('input', '')
        output = item.get('output', '')
        
        if input_text:
            prompt = f"{instruction}\n\n{input_text}"
        else:
            prompt = instruction
        
        # Qwen 格式
        return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>\n"
    
    def train(
        self,
        train_data_path: str,
        eval_data_path: Optional[str] = None
    ):
        """
        执行训练
        
        Args:
            train_data_path: 训练数据路径
            eval_data_path: 验证数据路径（可选）
        """
        if self.model is None:
            self.setup_model()
        
        # 准备数据集
        train_dataset = self.prepare_dataset(train_data_path)
        eval_dataset = self.prepare_dataset(eval_data_path) if eval_data_path else None
        
        # 训练参数
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_train_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            warmup_ratio=self.config.warmup_ratio,
            lr_scheduler_type=self.config.lr_scheduler_type,
            save_steps=self.config.save_steps,
            eval_steps=self.config.eval_steps if eval_dataset else None,
            evaluation_strategy="steps" if eval_dataset else "no",
            logging_steps=self.config.logging_steps,
            save_total_limit=self.config.save_total_limit,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            seed=self.config.seed,
            report_to="none",  # 可以改为 "wandb" 或 "tensorboard"
            load_best_model_at_end=True if eval_dataset else False
        )
        
        # 使用 SFTTrainer（如果可用）
        if HAS_TRL:
            self.trainer = SFTTrainer(
                model=self.model,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                args=training_args,
                dataset_text_field="text",
                max_seq_length=self.config.max_seq_length,
                packing=False
            )
        else:
            # 使用标准 Trainer
            def tokenize_function(examples):
                return self.tokenizer(
                    examples['text'],
                    truncation=True,
                    max_length=self.config.max_seq_length,
                    padding="max_length"
                )
            
            train_dataset = train_dataset.map(tokenize_function, batched=True)
            if eval_dataset:
                eval_dataset = eval_dataset.map(tokenize_function, batched=True)
            
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model,
                padding=True
            )
            
            self.trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                data_collator=data_collator
            )
        
        print("🚀 Starting training...")
        
        # 开始训练
        train_result = self.trainer.train()
        
        # 保存模型
        self.save_model()
        
        # 保存训练结果
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        
        print("✅ Training complete!")
        return metrics
    
    def save_model(self, output_dir: Optional[str] = None):
        """保存模型"""
        save_dir = output_dir or self.config.output_dir
        
        # 保存 LoRA 权重
        self.model.save_pretrained(save_dir)
        self.tokenizer.save_pretrained(save_dir)
        
        # 保存配置
        config_path = Path(save_dir) / "training_config.json"
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"💾 Model saved to {save_dir}")
    
    def load_model(self, model_path: str):
        """加载已训练的模型"""
        if not HAS_TRANSFORMERS or not HAS_PEFT:
            raise ImportError("transformers and peft are required")
        
        from peft import PeftModel
        
        print(f"🔄 Loading trained model from: {model_path}")
        
        # 加载基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16
        )
        
        # 加载 LoRA 权重
        self.model = PeftModel.from_pretrained(base_model, model_path)
        
        # 加载分词器
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        print("✅ Model loaded")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """生成文本"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded")
        
        # 格式化输入
        formatted_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # 提取助手回复
        if "<|im_start|>assistant\n" in response:
            response = response.split("<|im_start|>assistant\n")[-1]
        if "<|im_end|>" in response:
            response = response.split("<|im_end|>")[0]
        
        return response.strip()


def create_training_script(output_path: str, config: TrainingConfig):
    """生成独立的训练脚本"""
    script = f'''#!/usr/bin/env python
"""
LoRA Fine-tuning Script for Qwen2.5-7B
自动生成的训练脚本
"""

import sys
sys.path.insert(0, '{Path(__file__).parent.parent.parent}')

from optional.finetune import LoRATrainer, TrainingConfig

def main():
    # 训练配置
    config = TrainingConfig(
        model_name="{config.model_name}",
        lora_r={config.lora_r},
        lora_alpha={config.lora_alpha},
        lora_dropout={config.lora_dropout},
        use_4bit={config.use_4bit},
        output_dir="{config.output_dir}",
        num_train_epochs={config.num_train_epochs},
        per_device_train_batch_size={config.per_device_train_batch_size},
        gradient_accumulation_steps={config.gradient_accumulation_steps},
        learning_rate={config.learning_rate},
        max_seq_length={config.max_seq_length}
    )
    
    # 初始化训练器
    trainer = LoRATrainer(config)
    
    # 开始训练
    # 请修改数据路径
    trainer.train(
        train_data_path="data/training/train_chat.json",
        eval_data_path="data/training/val_chat.json"
    )

if __name__ == "__main__":
    main()
'''
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(script)
    
    print(f"📝 Training script saved to {output_path}")


if __name__ == "__main__":
    # 测试配置
    config = TrainingConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        output_dir="output/lora_qwen",
        num_train_epochs=3,
        per_device_train_batch_size=4
    )
    
    print("Training Config:")
    print(json.dumps(config.to_dict(), indent=2))
    
    # 生成训练脚本
    script_path = Path(__file__).parent.parent.parent / "scripts" / "train_lora.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    create_training_script(str(script_path), config)
