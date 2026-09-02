"""
Data Analyzer Tool
数据分析工具
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from pathlib import Path
import json


import logging
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """数据分析工具"""
    
    def __init__(self, config: dict):
        self.config = config
    
    def analyze_dataset(self, data_path: str) -> Dict:
        """
        分析数据集
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            分析结果
        """
        logger.info(f"📊 Analyzing dataset: {data_path}")
        
        try:
            # 加载数据
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            elif data_path.endswith('.json'):
                df = pd.read_json(data_path)
            else:
                return {"error": "Unsupported file format"}
            
            analysis = {
                'basic_info': self._get_basic_info(df),
                'statistics': self._get_statistics(df),
                'missing_values': self._analyze_missing_values(df),
                'data_types': self._analyze_data_types(df),
                'correlations': self._analyze_correlations(df)
            }
            
            return analysis
            
        except Exception as e:
            return {"error": str(e)}
    
    def _get_basic_info(self, df: pd.DataFrame) -> Dict:
        """获取基本信息"""
        return {
            'num_rows': len(df),
            'num_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage': f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        }
    
    def _get_statistics(self, df: pd.DataFrame) -> Dict:
        """获取统计信息"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return {}
        
        stats = df[numeric_cols].describe().to_dict()
        
        return stats
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict:
        """分析缺失值"""
        missing = df.isnull().sum()
        missing_percent = (missing / len(df) * 100).round(2)
        
        missing_info = {}
        for col in df.columns:
            if missing[col] > 0:
                missing_info[col] = {
                    'count': int(missing[col]),
                    'percentage': float(missing_percent[col])
                }
        
        return missing_info
    
    def _analyze_data_types(self, df: pd.DataFrame) -> Dict:
        """分析数据类型"""
        type_counts = df.dtypes.value_counts().to_dict()
        
        return {str(k): int(v) for k, v in type_counts.items()}
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict:
        """分析相关性"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) < 2:
            return {}
        
        corr_matrix = df[numeric_cols].corr()
        
        # 找出高相关性的特征对
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    high_corr.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': float(corr_value)
                    })
        
        return {
            'high_correlations': high_corr,
            'correlation_matrix_shape': corr_matrix.shape
        }
    
    def suggest_preprocessing(self, analysis: Dict) -> List[str]:
        """
        根据分析结果建议预处理步骤
        
        Args:
            analysis: 数据分析结果
            
        Returns:
            预处理建议列表
        """
        suggestions = []
        
        # 检查缺失值
        if analysis.get('missing_values'):
            suggestions.append("处理缺失值：考虑删除或填充")
        
        # 检查数据类型
        data_types = analysis.get('data_types', {})
        if 'object' in data_types:
            suggestions.append("编码分类变量：使用one-hot或label encoding")
        
        # 检查相关性
        high_corr = analysis.get('correlations', {}).get('high_correlations', [])
        if high_corr:
            suggestions.append(f"处理高相关性特征：发现{len(high_corr)}对高相关特征")
        
        # 检查数据规模
        num_rows = analysis.get('basic_info', {}).get('num_rows', 0)
        if num_rows < 100:
            suggestions.append("数据量较小：考虑数据增强或使用简单模型")
        
        # 建议标准化
        suggestions.append("特征标准化：使用StandardScaler或MinMaxScaler")
        
        return suggestions
    
    def compare_datasets(self, data_paths: List[str]) -> Dict:
        """
        比较多个数据集
        
        Args:
            data_paths: 数据文件路径列表
            
        Returns:
            比较结果
        """
        comparisons = {}
        
        for path in data_paths:
            analysis = self.analyze_dataset(path)
            comparisons[Path(path).name] = analysis.get('basic_info', {})
        
        return comparisons
    
    def detect_outliers(self, data_path: str, method: str = 'iqr') -> Dict:
        """
        检测异常值
        
        Args:
            data_path: 数据文件路径
            method: 检测方法 ('iqr', 'zscore')
            
        Returns:
            异常值检测结果
        """
        try:
            df = pd.read_csv(data_path)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            outliers = {}
            
            for col in numeric_cols:
                if method == 'iqr':
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    
                elif method == 'zscore':
                    z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                    outlier_mask = z_scores > 3
                
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float(outlier_count / len(df) * 100)
                    }
            
            return outliers
            
        except Exception as e:
            return {"error": str(e)}
    
    def generate_report(self, analysis: Dict, output_path: str):
        """
        生成分析报告
        
        Args:
            analysis: 分析结果
            output_path: 输出文件路径
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 生成Markdown报告
            report = "# 数据分析报告\n\n"
            
            # 基本信息
            report += "## 基本信息\n\n"
            basic_info = analysis.get('basic_info', {})
            for key, value in basic_info.items():
                report += f"- **{key}**: {value}\n"
            
            # 统计信息
            report += "\n## 统计信息\n\n"
            stats = analysis.get('statistics', {})
            if stats:
                report += "| 特征 | 均值 | 标准差 | 最小值 | 最大值 |\n"
                report += "|------|------|--------|--------|--------|\n"
                for col, col_stats in stats.items():
                    report += f"| {col} | {col_stats.get('mean', 0):.2f} | "
                    report += f"{col_stats.get('std', 0):.2f} | "
                    report += f"{col_stats.get('min', 0):.2f} | "
                    report += f"{col_stats.get('max', 0):.2f} |\n"
            
            # 缺失值
            report += "\n## 缺失值分析\n\n"
            missing = analysis.get('missing_values', {})
            if missing:
                for col, info in missing.items():
                    report += f"- **{col}**: {info['count']} ({info['percentage']}%)\n"
            else:
                report += "无缺失值\n"
            
            # 预处理建议
            report += "\n## 预处理建议\n\n"
            suggestions = self.suggest_preprocessing(analysis)
            for i, suggestion in enumerate(suggestions, 1):
                report += f"{i}. {suggestion}\n"
            
            # 保存报告
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"📄 Report saved to {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Error generating report: {e}")


if __name__ == "__main__":
    config = {}
    analyzer = DataAnalyzer(config)
    
    # 测试（需要实际数据文件）
    # analysis = analyzer.analyze_dataset('data.csv')
    # print(json.dumps(analysis, indent=2))
