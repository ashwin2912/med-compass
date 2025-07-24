#!/usr/bin/env python3
"""
Topic modeling using NMF for clinical text analysis
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class TopicResult:
    """Container for topic modeling results"""
    topic_id: int
    keywords: List[str]
    weights: List[float]
    description: str = ""

@dataclass
class DocumentTopics:
    """Container for document-topic associations"""
    document_id: int
    topic_distributions: Dict[int, float]
    dominant_topic: int
    dominant_topic_weight: float

class ClinicalTopicModeler:
    """NMF-based topic modeling for clinical text"""
    
    def __init__(self, 
                 n_topics: int = 5,
                 max_df: float = 0.95,
                 min_df: int = 2,
                 ngram_range: Tuple[int, int] = (1, 2),
                 max_features: Optional[int] = 1000):
        """
        Initialize topic modeler
        
        Args:
            n_topics: Number of topics to extract
            max_df: Maximum document frequency for terms
            min_df: Minimum document frequency for terms
            ngram_range: Range of n-grams to extract
            max_features: Maximum number of features to use
        """
        self.n_topics = n_topics
        self.max_df = max_df
        self.min_df = min_df
        self.ngram_range = ngram_range
        self.max_features = max_features
        
        # Clinical stopwords in addition to standard English stopwords
        self.clinical_stopwords = {
            'patient', 'hospital', 'discharge', 'admission', 'day', 'time',
            'given', 'taken', 'noted', 'showed', 'history', 'exam', 'findings',
            'stable', 'continue', 'follow', 'appointment', 'clinic', 'unit'
        }
        
        self.vectorizer = None
        self.nmf_model = None
        self.feature_names = None
        self.topics = []
        self.is_fitted = False
    
    def _create_vectorizer(self) -> TfidfVectorizer:
        """Create TF-IDF vectorizer with clinical text optimizations"""
        # Combine standard English stopwords with clinical ones
        all_stopwords = list(self.clinical_stopwords) + ['english']
        
        return TfidfVectorizer(
            stop_words=all_stopwords,
            max_df=self.max_df,
            min_df=self.min_df,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            lowercase=True,
            token_pattern=r'\b[a-zA-Z][a-zA-Z]+\b'  # Only alphabetic tokens
        )
    
    def fit(self, documents: List[str]) -> 'ClinicalTopicModeler':
        """
        Fit the topic model on documents
        
        Args:
            documents: List of clinical text documents
            
        Returns:
            Self for method chaining
        """
        if len(documents) < self.n_topics:
            raise ValueError(f"Need at least {self.n_topics} documents for {self.n_topics} topics")
        
        # Create and fit vectorizer
        self.vectorizer = self._create_vectorizer()
        X = self.vectorizer.fit_transform(documents)
        self.feature_names = self.vectorizer.get_feature_names_out()
        
        # Fit NMF model
        self.nmf_model = NMF(
            n_components=self.n_topics,
            random_state=42,
            max_iter=200,
            alpha_W=0.1,  # Regularization for sparsity
            alpha_H=0.1
        )
        
        W = self.nmf_model.fit_transform(X)  # document-topic matrix
        H = self.nmf_model.components_       # topic-word matrix
        
        # Extract topics
        self.topics = self._extract_topics(H)
        self.is_fitted = True
        
        return self
    
    def _extract_topics(self, H: np.ndarray, top_n: int = 10) -> List[TopicResult]:
        """Extract top keywords for each topic"""
        topics = []
        
        for topic_idx, topic_weights in enumerate(H):
            # Get top features for this topic
            top_indices = topic_weights.argsort()[:-top_n-1:-1]
            keywords = [self.feature_names[i] for i in top_indices]
            weights = [topic_weights[i] for i in top_indices]
            
            # Create topic description from top 3 keywords
            description = f"Topic about {', '.join(keywords[:3])}"
            
            topics.append(TopicResult(
                topic_id=topic_idx,
                keywords=keywords,
                weights=weights,
                description=description
            ))
        
        return topics
    
    def transform(self, documents: List[str]) -> List[DocumentTopics]:
        """
        Get topic distributions for documents
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            List of DocumentTopics with topic distributions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before transform")
        
        # Transform documents
        X = self.vectorizer.transform(documents)
        W = self.nmf_model.transform(X)
        
        results = []
        for doc_idx, doc_topics in enumerate(W):
            # Normalize to get probabilities
            doc_topics_norm = doc_topics / doc_topics.sum()
            
            # Create topic distribution dict
            topic_dist = {i: float(weight) for i, weight in enumerate(doc_topics_norm)}
            
            # Find dominant topic
            dominant_topic = int(np.argmax(doc_topics_norm))
            dominant_weight = float(doc_topics_norm[dominant_topic])
            
            results.append(DocumentTopics(
                document_id=doc_idx,
                topic_distributions=topic_dist,
                dominant_topic=dominant_topic,
                dominant_topic_weight=dominant_weight
            ))
        
        return results
    
    def fit_transform(self, documents: List[str]) -> Tuple[List[TopicResult], List[DocumentTopics]]:
        """Fit model and transform documents in one step"""
        self.fit(documents)
        doc_topics = self.transform(documents)
        return self.topics, doc_topics
    
    def get_topic_summary(self) -> pd.DataFrame:
        """Get a summary of all topics as DataFrame"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        summary_data = []
        for topic in self.topics:
            summary_data.append({
                'Topic_ID': topic.topic_id,
                'Description': topic.description,
                'Top_Keywords': ', '.join(topic.keywords[:5]),
                'Top_Weights': ', '.join([f"{w:.3f}" for w in topic.weights[:5]])
            })
        
        return pd.DataFrame(summary_data)
    
    def print_topics(self, top_n: int = 10):
        """Print topics in a readable format"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        print("🔍 Discovered Clinical Topics")
        print("=" * 60)
        
        for topic in self.topics:
            print(f"\n📋 Topic #{topic.topic_id + 1}: {topic.description}")
            print("-" * 40)
            
            for i, (keyword, weight) in enumerate(zip(topic.keywords[:top_n], topic.weights[:top_n])):
                print(f"  {i+1:2d}. {keyword:<20} (weight: {weight:.3f})")

def analyze_clinical_topics(documents: List[str], n_topics: int = 5) -> Tuple[ClinicalTopicModeler, List[DocumentTopics]]:
    """
    Convenience function to perform topic modeling on clinical documents
    
    Args:
        documents: List of clinical text documents
        n_topics: Number of topics to extract
        
    Returns:
        Fitted topic modeler and document-topic associations
    """
    modeler = ClinicalTopicModeler(n_topics=n_topics)
    topics, doc_topics = modeler.fit_transform(documents)
    
    return modeler, doc_topics