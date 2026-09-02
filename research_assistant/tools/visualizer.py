"""
Visualizer Tool
数据可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


import logging
logger = logging.getLogger(__name__)

class Visualizer:
    """可视化工具"""
    
    def __init__(self, config: dict):
        self.config = config
        
        # 设置样式
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def plot_training_history(
        self,
        history: Dict,
        output_path: str,
        metrics: Optional[List[str]] = None
    ):
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            output_path: 输出路径
            metrics: 要绘制的指标列表
        """
        if metrics is None:
            metrics = list(history.keys())
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
        
        if n_metrics == 1:
            axes = [axes]
        
        for ax, metric in zip(axes, metrics):
            if metric in history:
                ax.plot(history[metric], label=f'Train {metric}')
                
                # 如果有验证数据
                val_metric = f'val_{metric}'
                if val_metric in history:
                    ax.plot(history[val_metric], label=f'Val {metric}')
                
                ax.set_xlabel('Epoch')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} over Epochs')
                ax.legend()
                ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Training history plot saved to {output_path}")
    
    def plot_predictions(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        output_path: str,
        title: str = "Predictions vs Actual"
    ):
        """
        绘制预测vs实际值
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            output_path: 输出路径
            title: 图表标题
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 散点图
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title(title)
        axes[0].grid(True)
        
        # 残差图
        residuals = y_true - y_pred
        axes[1].scatter(y_pred, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot')
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Predictions plot saved to {output_path}")
    
    def plot_feature_importance(
        self,
        features: List[str],
        importance: np.ndarray,
        output_path: str,
        top_k: int = 20
    ):
        """
        绘制特征重要性
        
        Args:
            features: 特征名称列表
            importance: 重要性分数
            output_path: 输出路径
            top_k: 显示top-k个特征
        """
        # 排序
        indices = np.argsort(importance)[::-1][:top_k]
        
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(indices)), importance[indices])
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Importance Score')
        plt.title(f'Top {top_k} Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Feature importance plot saved to {output_path}")
    
    def plot_correlation_matrix(
        self,
        data: pd.DataFrame,
        output_path: str,
        method: str = 'pearson'
    ):
        """
        绘制相关性矩阵热图
        
        Args:
            data: 数据DataFrame
            output_path: 输出路径
            method: 相关性计算方法
        """
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr(method=method)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0,
                   square=True, linewidths=1)
        plt.title(f'Correlation Matrix ({method.capitalize()})')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Correlation matrix saved to {output_path}")
    
    def plot_distribution(
        self,
        data: pd.DataFrame,
        columns: List[str],
        output_path: str
    ):
        """
        绘制数据分布
        
        Args:
            data: 数据DataFrame
            columns: 要绘制的列
            output_path: 输出路径
        """
        n_cols = len(columns)
        n_rows = (n_cols + 2) // 3
        
        fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5*n_rows))
        axes = axes.flatten() if n_cols > 1 else [axes]
        
        for i, col in enumerate(columns):
            if col in data.columns:
                axes[i].hist(data[col].dropna(), bins=50, edgecolor='black')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
                axes[i].set_title(f'Distribution of {col}')
                axes[i].grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_cols, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Distribution plot saved to {output_path}")
    
    def plot_time_series(
        self,
        data: pd.DataFrame,
        time_col: str,
        value_cols: List[str],
        output_path: str
    ):
        """
        绘制时间序列
        
        Args:
            data: 数据DataFrame
            time_col: 时间列名
            value_cols: 值列名列表
            output_path: 输出路径
        """
        plt.figure(figsize=(12, 6))
        
        for col in value_cols:
            if col in data.columns:
                plt.plot(data[time_col], data[col], label=col, marker='o')
        
        plt.xlabel(time_col)
        plt.ylabel('Value')
        plt.title('Time Series Plot')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Time series plot saved to {output_path}")
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: List[str],
        output_path: str
    ):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 标签名称
            output_path: 输出路径
        """
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Confusion matrix saved to {output_path}")
    
    def create_experiment_dashboard(
        self,
        results: Dict,
        output_path: str
    ):
        """
        创建实验结果仪表板
        
        Args:
            results: 实验结果字典
            output_path: 输出路径
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 这里可以添加多个子图展示不同的实验结果
        # 示例：训练曲线、性能对比、参数影响等
        
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"📊 Experiment dashboard saved to {output_path}")


if __name__ == "__main__":
    config = {}
    viz = Visualizer(config)
    
    # 测试绘制训练历史
    history = {
        'loss': [0.5, 0.4, 0.3, 0.2, 0.1],
        'val_loss': [0.6, 0.5, 0.4, 0.3, 0.25]
    }
    
    # viz.plot_training_history(history, 'training_history.png')
