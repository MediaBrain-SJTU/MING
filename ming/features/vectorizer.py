"""
医疗文本向量化模块

提供医疗文本的向量化表示功能，支持多种向量化策略。
"""

import hashlib
from typing import Dict, List, Optional, Union, Any
import numpy as np
from .feature_extractor import ExtractedFeatures, FeatureExtractor


class MedicalVectorizer:
    """
    医疗文本向量化器
    
    将医疗文本转换为数值向量，支持特征工程向量、
    TF-IDF向量等多种表示方式。
    
    Attributes:
        feature_extractor: 特征提取器
        vector_dim: 向量维度
    
    Example:
        >>> vectorizer = MedicalVectorizer()
        >>> vector = vectorizer.vectorize("患者有高血压病史")
    """
    
    def __init__(
        self, 
        feature_extractor: Optional[FeatureExtractor] = None,
        use_feature_engineering: bool = True,
        vector_dim: int = 128
    ):
        """
        初始化向量化器
        
        Args:
            feature_extractor: 特征提取器实例
            use_feature_engineering: 是否使用特征工程
            vector_dim: 向量维度
        """
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.use_feature_engineering = use_feature_engineering
        self.vector_dim = vector_dim
    
    def vectorize(self, text: str) -> np.ndarray:
        """
        将文本转换为向量
        
        Args:
            text: 输入文本
            
        Returns:
            向量表示
        """
        if self.use_feature_engineering:
            features = self.feature_extractor.extract(text)
            vector = features.to_vector()
            
            # 填充或截断到指定维度
            if len(vector) < self.vector_dim:
                vector = np.pad(
                    vector, 
                    (0, self.vector_dim - len(vector)), 
                    mode='constant'
                )
            elif len(vector) > self.vector_dim:
                vector = vector[:self.vector_dim]
            
            return vector
        else:
            # 使用简单的哈希向量化作为fallback
            return self._hash_vectorize(text)
    
    def batch_vectorize(self, texts: List[str]) -> np.ndarray:
        """
        批量向量化文本
        
        Args:
            texts: 文本列表
            
        Returns:
            向量矩阵
        """
        vectors = [self.vectorize(text) for text in texts]
        return np.array(vectors)
    
    def _hash_vectorize(self, text: str) -> np.ndarray:
        """使用哈希进行简单向量化"""
        vector = np.zeros(self.vector_dim)
        
        # 使用多个哈希函数
        for i, char in enumerate(text):
            hash_val = int(hashlib.md5(char.encode()).hexdigest(), 16)
            idx = hash_val % self.vector_dim
            vector[idx] += 1.0
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def compute_similarity(
        self, 
        text1: str, 
        text2: str
    ) -> float:
        """
        计算两个文本的相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            余弦相似度
        """
        vec1 = self.vectorize(text1)
        vec2 = self.vectorize(text2)
        
        return self._cosine_similarity(vec1, vec2)
    
    def _cosine_similarity(
        self, 
        vec1: np.ndarray, 
        vec2: np.ndarray
    ) -> float:
        """计算余弦相似度"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def find_similar_texts(
        self,
        query: str,
        candidates: List[str],
        top_k: int = 5
    ) -> List[tuple]:
        """
        查找与查询文本最相似的文本
        
        Args:
            query: 查询文本
            candidates: 候选文本列表
            top_k: 返回最相似的数量
            
        Returns:
            (索引, 相似度) 元组列表
        """
        query_vec = self.vectorize(query)
        candidate_vecs = self.batch_vectorize(candidates)
        
        # 计算相似度
        similarities = [
            self._cosine_similarity(query_vec, cand_vec)
            for cand_vec in candidate_vecs
        ]
        
        # 获取top-k
        indexed_sims = list(enumerate(similarities))
        indexed_sims.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_sims[:top_k]
