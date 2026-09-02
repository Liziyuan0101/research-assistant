"""
Query Enhancement Utilities
查询增强工具：自动将用户查询转换为学术检索关键词
支持 KeyBERT 进行语义关键短语提取
"""

import re
from typing import Dict, List, Optional, Tuple

# KeyBERT for semantic keyphrase extraction
try:
    from keybert import KeyBERT
    HAS_KEYBERT = True
except ImportError:
    HAS_KEYBERT = False
    KeyBERT = None


import logging
logger = logging.getLogger(__name__)

class QueryEnhancer:
    """查询增强器 - 支持 KeyBERT 语义短语提取"""
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        
        # 中英文术语映射表
        self.term_mapping = {
            # 深度学习相关
            '深度学习': 'deep learning',
            '机器学习': 'machine learning',
            '神经网络': 'neural network',
            '图神经网络': 'graph neural network',
            'GNN': 'graph neural network',
            '卷积神经网络': 'convolutional neural network',
            'CNN': 'convolutional neural network',
            '循环神经网络': 'recurrent neural network',
            'RNN': 'recurrent neural network',
            '长短期记忆': 'LSTM',
            '注意力机制': 'attention mechanism',
            'Transformer': 'transformer',
            
            # 电池相关
            '电池的健康状态': 'battery state-of-health',
            '电池健康状态': 'battery state-of-health',
            '健康状态': 'state-of-health',
            '电池健康': 'battery health',
            '电池': 'battery',
            '锂电池': 'lithium battery',
            '锂离子电池': 'lithium-ion battery',
            'SOH': 'state-of-health',
            '剩余寿命': 'remaining-useful-life',
            '剩余使用寿命': 'remaining-useful-life',
            'RUL': 'remaining-useful-life',
            '寿命预测': 'life prediction',
            '容量衰减': 'capacity degradation',
            '电池老化': 'battery aging',
            '充电状态': 'state-of-charge',
            'SOC': 'state-of-charge',
            
            # 预测相关
            '预测': 'prediction',
            '预报': 'forecasting',
            '估计': 'estimation',
            '诊断': 'diagnosis',
            '故障检测': 'fault detection',
            '异常检测': 'anomaly detection',
            
            # 方法相关
            '时间序列': 'time series',
            '特征提取': 'feature extraction',
            '数据驱动': 'data-driven',
            '迁移学习': 'transfer learning',
            '强化学习': 'reinforcement learning',
            '监督学习': 'supervised learning',
            '无监督学习': 'unsupervised learning',
            
            # 应用领域
            '电动汽车': 'electric vehicle',
            '储能': 'energy storage',
            '可再生能源': 'renewable energy',
        }
        
        # 关键词扩展规则
        self.expansion_rules = {
            'graph neural network': ['GNN', 'graph convolutional network', 'GCN', 'graph attention network', 'GAT'],
            'battery': ['lithium-ion battery', 'Li-ion battery', 'battery cell'],
            'state-of-health': ['SOH', 'battery health', 'health estimation'],
            'remaining-useful-life': ['RUL', 'life prediction', 'lifetime prediction', 'prognostics'],
            'prediction': ['forecasting', 'estimation', 'prognosis'],
            'deep learning': ['neural network', 'machine learning', 'artificial intelligence'],
        }
        
        # KeyBERT 模型（延迟加载）
        self._keybert_model = None
        self.use_keybert = config.get('use_keybert', True) if config else True
    
    @property
    def keybert_model(self):
        """延迟加载 KeyBERT 模型"""
        if self._keybert_model is None and HAS_KEYBERT and self.use_keybert:
            self._keybert_model = KeyBERT(model='all-MiniLM-L6-v2')
        return self._keybert_model
    
    def extract_keyphrases(
        self, 
        query: str, 
        top_n: int = 5,
        keyphrase_ngram_range: Tuple[int, int] = (1, 3),
        use_mmr: bool = True,
        diversity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        使用 KeyBERT 提取关键短语
        
        Args:
            query: 输入文本
            top_n: 返回的关键短语数量
            keyphrase_ngram_range: n-gram 范围，(1,3) 表示提取 1-3 个词的短语
            use_mmr: 是否使用 MMR 算法增加多样性
            diversity: 多样性参数 (0-1)
            
        Returns:
            关键短语列表，每个元素为 (短语, 分数)
        """
        if not HAS_KEYBERT or self.keybert_model is None:
            # 回退到简单分词
            return [(term, 1.0) for term in self._extract_key_terms_simple(query)]
        
        try:
            keywords = self.keybert_model.extract_keywords(
                query,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words='english',
                top_n=top_n,
                use_mmr=use_mmr,
                diversity=diversity
            )
            return keywords
        except Exception as e:
            logger.warning(f"⚠️ KeyBERT extraction failed: {e}")
            return [(term, 1.0) for term in self._extract_key_terms_simple(query)]
    
    def _extract_key_terms_simple(self, query: str) -> List[str]:
        """简单的关键词提取（回退方法）"""
        words = query.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
                      'using', 'how', 'what', 'which', 'when', 'where', 'why'}
        return [word for word in words if word.lower() not in stop_words and len(word) > 2][:5]
    
    def enhance_query(self, query: str, mode: str = 'auto') -> str:
        """
        增强查询
        
        Args:
            query: 原始查询（中文或英文）
            mode: 模式 ('auto', 'translate', 'expand')
            
        Returns:
            增强后的英文查询
        """
        # 检测是否包含中文
        has_chinese = bool(re.search(r'[\u4e00-\u9fff]', query))
        
        if has_chinese:
            # 翻译中文查询
            english_query = self._translate_to_english(query)
        else:
            english_query = query
        
        # 清理和标准化
        english_query = self._normalize_query(english_query)
        
        return english_query
    
    def _translate_to_english(self, chinese_query: str) -> str:
        """
        将中文查询翻译为英文学术术语
        
        Args:
            chinese_query: 中文查询
            
        Returns:
            英文查询
        """
        result = chinese_query
        
        # 移除常见的中文疑问词和连接词（先移除）
        remove_words = ['如何', '怎么', '怎样', '什么', '哪些', '使用', '进行', '？', '?', '、', '，']
        for word in remove_words:
            result = result.replace(word, ' ')
        
        # 按长度排序，优先匹配长词组
        sorted_terms = sorted(self.term_mapping.items(), key=lambda x: len(x[0]), reverse=True)
        
        for chinese, english in sorted_terms:
            # 替换时在两边加空格，避免粘连
            result = result.replace(chinese, f' {english} ')
        
        # 移除常见的中文连接词和助词
        chinese_particles = ['的', '中', '在', '与', '和', '及', '对', '为', '以', '从', 
                            '应用', '方法', '研究', '分析', '技术', '系统', '基于', '通过']
        for particle in chinese_particles:
            result = result.replace(particle, ' ')
        
        # 移除所有剩余的中文字符
        result = re.sub(r'[\u4e00-\u9fff]+', ' ', result)
        
        # 清理多余空格
        result = ' '.join(result.split())
        
        # 转小写
        result = result.lower()
        
        return result
    
    def _normalize_query(self, query: str) -> str:
        """标准化查询"""
        # 转小写
        query = query.lower()
        
        # 移除特殊字符（保留空格、字母数字和连字符）
        query = re.sub(r'[^\w\s-]', ' ', query)
        
        # 将连字符替换为空格（用于arXiv等搜索引擎）
        # 但保留完整术语的连字符
        # query = query.replace('-', ' ')
        
        # 清理多余空格
        query = ' '.join(query.split())
        
        return query
    
    def expand_query(self, query: str, max_terms: int = 3) -> List[str]:
        """
        扩展查询，生成相关术语
        
        Args:
            query: 查询
            max_terms: 最大扩展术语数
            
        Returns:
            扩展后的术语列表
        """
        expanded = [query]
        
        for key_term, expansions in self.expansion_rules.items():
            if key_term in query.lower():
                expanded.extend(expansions[:max_terms])
        
        return list(set(expanded))[:5]  # 返回前5个不重复的术语
    
    def suggest_search_keywords(self, query: str, broad_search: bool = False) -> Dict[str, any]:
        """
        为查询建议搜索关键词
        
        Args:
            query: 用户查询
            broad_search: 是否使用更广泛的搜索（提高召回率）
            
        Returns:
            包含主查询和扩展信息的字典
        """
        # 增强查询
        main_query = self.enhance_query(query)
        
        # 扩展查询
        expanded_queries = self.expand_query(main_query)
        
        # 提取关键术语
        key_terms = self._extract_key_terms(main_query)
        
        # 如果启用广泛搜索，使用更通用的关键词
        if broad_search:
            # 移除过于具体的词，保留核心概念
            core_terms = []
            for term in key_terms:
                if term in ['battery', 'prediction', 'estimation', 'health', 'neural', 'network', 
                           'deep', 'learning', 'machine', 'state', 'life', 'remaining']:
                    core_terms.append(term)
            
            # 生成更广泛的查询
            if len(core_terms) >= 2:
                broad_query = ' '.join(core_terms[:3])  # 只用前3个核心词
            else:
                broad_query = main_query
            
            suggested_query = broad_query
        else:
            suggested_query = main_query
        
        return {
            'original_query': query,
            'main_query': main_query,
            'expanded_queries': expanded_queries,
            'key_terms': key_terms,
            'suggested_query': suggested_query,
            'broad_search': broad_search
        }
    
    def _extract_key_terms(self, query: str, use_keybert: bool = True) -> List[str]:
        """
        提取关键术语
        
        Args:
            query: 查询文本
            use_keybert: 是否使用 KeyBERT（默认 True）
            
        Returns:
            关键术语列表
        """
        # 优先使用 KeyBERT 进行语义短语提取
        if use_keybert and HAS_KEYBERT and self.use_keybert:
            keyphrases = self.extract_keyphrases(
                query, 
                top_n=5, 
                keyphrase_ngram_range=(1, 3)
            )
            # 返回短语列表（不包含分数）
            return [phrase for phrase, score in keyphrases]
        
        # 回退到简单分词
        return self._extract_key_terms_simple(query)


if __name__ == "__main__":
    # 测试
    enhancer = QueryEnhancer()
    
    test_queries = [
        "如何使用图神经网络预测电池剩余寿命？",
        "深度学习在电池健康状态估计中的应用",
        "锂离子电池容量衰减预测方法",
        "battery state of health prediction",
        "graph neural network for battery prediction"
    ]
    
    print("=" * 60)
    print("查询增强测试 (KeyBERT 语义短语提取)")
    print("=" * 60 + "\n")
    
    for query in test_queries:
        print(f"原始查询: {query}")
        
        # 使用 KeyBERT 提取关键短语
        keyphrases = enhancer.extract_keyphrases(query)
        print(f"KeyBERT 关键短语:")
        for phrase, score in keyphrases:
            print(f"  - {phrase}: {score:.4f}")
        
        # 完整的搜索建议
        result = enhancer.suggest_search_keywords(query)
        print(f"增强查询: {result['suggested_query']}")
        print(f"关键术语: {', '.join(result['key_terms'])}")
        print("-" * 60 + "\n")
