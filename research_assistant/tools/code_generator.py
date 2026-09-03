"""
Code Generator Tool
自动生成实验代码的工具
"""

import os
from typing import Dict, Optional, List
from pathlib import Path

from ..utils.llm import create_openai_client, chat_completion


import logging
logger = logging.getLogger(__name__)

class CodeGenerator:
    """代码生成工具"""
    
    def __init__(self, config: dict):
        self.config = config
        
        llm_config = config.get('llm', {})
        
        self.client = create_openai_client(config)
        
        self.model = llm_config.get('model', 'deepseek-chat')
    
    def generate_training_code(
        self,
        model_type: str,
        dataset_info: Dict,
        parameters: Dict,
        framework: str = "pytorch"
    ) -> str:
        """
        生成训练代码
        
        Args:
            model_type: 模型类型
            dataset_info: 数据集信息
            parameters: 训练参数
            framework: 框架
            
        Returns:
            生成的代码
        """
        prompt = f"""
请生成完整的{framework}训练代码：

模型类型: {model_type}
数据集信息: {dataset_info}
训练参数: {parameters}

要求：
1. 包含数据加载、预处理
2. 模型定义
3. 训练循环（包含验证）
4. 早停机制
5. 模型保存
6. 日志记录
7. 代码可直接运行

请生成完整的Python代码。
"""
        
        if not self.client:
            return self._template_training_code(model_type, framework)
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一位专业的{framework}开发者。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=8192
            )
            
            code = response.choices[0].message.content.strip()
            
            # 提取代码块
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            return code
            
        except Exception as e:
            logger.error(f"❌ Error generating training code: {e}")
            return self._template_training_code(model_type, framework)
    
    def generate_evaluation_code(
        self,
        model_type: str,
        metrics: List[str],
        framework: str = "pytorch"
    ) -> str:
        """生成评估代码"""
        prompt = f"""
请生成{framework}模型评估代码：

模型类型: {model_type}
评估指标: {', '.join(metrics)}

要求：
1. 加载训练好的模型
2. 在测试集上评估
3. 计算所有指标
4. 生成评估报告
5. 保存结果

请生成完整的Python代码。
"""
        
        if not self.client:
            return self._template_evaluation_code(metrics, framework)
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": f"你是一位专业的{framework}开发者。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4096
            )
            
            code = response.choices[0].message.content.strip()
            
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            return code
            
        except Exception as e:
            logger.error(f"❌ Error generating evaluation code: {e}")
            return self._template_evaluation_code(metrics, framework)
    
    def generate_data_preprocessing_code(
        self,
        data_description: str,
        preprocessing_steps: List[str]
    ) -> str:
        """生成数据预处理代码"""
        prompt = f"""
请生成数据预处理代码：

数据描述: {data_description}
预处理步骤: {', '.join(preprocessing_steps)}

要求：
1. 数据加载
2. 数据清洗
3. 特征工程
4. 数据标准化/归一化
5. 数据划分（训练/验证/测试）
6. 保存处理后的数据

请生成完整的Python代码。
"""
        
        if not self.client:
            return self._template_preprocessing_code()
        
        try:
            response = chat_completion(self.client, 
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一位数据科学专家。"
                    },
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=4096
            )
            
            code = response.choices[0].message.content.strip()
            
            if "```python" in code:
                code = code.split("```python")[1].split("```")[0].strip()
            elif "```" in code:
                code = code.split("```")[1].split("```")[0].strip()
            
            return code
            
        except Exception as e:
            logger.error(f"❌ Error generating preprocessing code: {e}")
            return self._template_preprocessing_code()
    
    def _template_training_code(self, model_type: str, framework: str) -> str:
        """训练代码模板"""
        return f"""
# Training Code for {model_type} using {framework}

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from tqdm import tqdm

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # TODO: Define model architecture
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        pass

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_data, batch_labels in tqdm(dataloader):
        batch_data = batch_data.to(device)
        batch_labels = batch_labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch_data, batch_labels in dataloader:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    epochs = 100
    batch_size = 32
    learning_rate = 0.001
    
    # TODO: Load data
    train_data, train_labels = None, None
    val_data, val_labels = None, None
    
    # Create datasets and dataloaders
    train_dataset = CustomDataset(train_data, train_labels)
    val_dataset = CustomDataset(val_data, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = Model().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = validate(model, val_loader, criterion, device)
        
        logger.info(f'Epoch {{epoch+1}}/{{epochs}}: Train Loss={{train_loss:.4f}}, Val Loss={{val_loss:.4f}}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info('Early stopping triggered')
                break

if __name__ == '__main__':
    main()
"""
    
    def _template_evaluation_code(self, metrics: List[str], framework: str) -> str:
        """评估代码模板"""
        metrics_code = ''.join([f"metrics['{m}'] = calculate_{m}(actuals, predictions)\n    " for m in metrics])
        return f"""
# Evaluation Code using {framework}

import torch
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            batch_data = batch_data.to(device)
            outputs = model(batch_data)
            
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_labels.numpy())
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    metrics = {{}}
    {metrics_code}
    
    return metrics

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = Model().to(device)
    model.load_state_dict(torch.load('best_model.pth'))
    
    # TODO: Load test data
    test_loader = None
    
    # Evaluate
    results = evaluate_model(model, test_loader, device)
    
    print("Evaluation Results:")
    for metric, value in results.items():
        print(f"{{metric}}: {{value:.4f}}")

if __name__ == '__main__':
    main()
"""
    
    def _template_preprocessing_code(self) -> str:
        """预处理代码模板"""
        return """
# Data Preprocessing Code

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    # TODO: Implement data loading
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    # Remove missing values
    df = df.dropna()
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    return df

def feature_engineering(df):
    # TODO: Add feature engineering steps
    return df

def preprocess_data(filepath, test_size=0.2, val_size=0.1):
    # Load data
    df = load_data(filepath)
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Split features and labels
    X = df.drop('target', axis=1).values
    y = df['target'].values
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(test_size + val_size), random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_size/(test_size + val_size), random_state=42
    )
    
    # Normalize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Save processed data
    np.save('X_train.npy', X_train)
    np.save('X_val.npy', X_val)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_val.npy', y_val)
    np.save('y_test.npy', y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == '__main__':
    preprocess_data('data.csv')
"""


if __name__ == "__main__":
    config = {'llm': {'model': 'gpt-4-turbo-preview'}}
    generator = CodeGenerator(config)
    
    code = generator.generate_training_code(
        model_type="GNN",
        dataset_info={"name": "battery_data", "size": 1000},
        parameters={"learning_rate": 0.001, "batch_size": 32},
        framework="pytorch"
    )
    
    print(code)
