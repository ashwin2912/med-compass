#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Data Preprocessors
Data cleaning and preprocessing with clinical domain knowledge
"""

import pandas as pd
import numpy as np
import re
import logging
from typing import List, Dict, Any, Set, Tuple, Optional
from datetime import datetime
from collections import Counter

from .models import ClinicalNote, DatasetConfig, DataQuality
from .exceptions import PreprocessingError

logger = logging.getLogger(__name__)


class BasePreprocessor:
    """
    Base class for data preprocessors
    
    Provides common functionality for data cleaning and transformation.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Dataset configuration object
        """
        self.config = config
        self.processing_stats = {
            'rows_input': 0,
            'rows_output': 0,
            'rows_filtered': 0,
            'duplicates_removed': 0,
            'data_cleaning_applied': 0,
            'type_conversions': 0
        }
    
    def preprocess(self, df: pd.DataFrame) -> List[ClinicalNote]:
        """
        Main preprocessing pipeline
        
        Args:
            df: Input DataFrame to preprocess
            
        Returns:
            List of processed ClinicalNote objects
            
        Raises:
            PreprocessingError: If preprocessing fails
        """
        try:
            logger.info("Starting data preprocessing pipeline...")
            self.processing_stats['rows_input'] = len(df)
            
            # Step 1: Map column names to standard format
            df_mapped = self._map_columns(df)
            
            # Step 2: Clean data types and handle missing values
            df_clean = self._clean_data_types(df_mapped)
            
            # Step 3: Apply filtering rules
            df_filtered = self._apply_filters(df_clean)
            
            # Step 4: Remove duplicates if enabled
            if self.config.deduplicate:
                df_deduped = self._remove_duplicates(df_filtered)
            else:
                df_deduped = df_filtered
            
            # Step 5: Apply sampling if configured
            df_sampled = self._apply_sampling(df_deduped)
            
            # Step 6: Convert to ClinicalNote objects
            notes = self._convert_to_notes(df_sampled)
            
            self.processing_stats['rows_output'] = len(notes)
            self.processing_stats['rows_filtered'] = self.processing_stats['rows_input'] - len(df_filtered)
            
            logger.info(f"Preprocessing complete: {self.processing_stats['rows_input']} → {len(notes)} notes")
            return notes
            
        except Exception as e:
            error_msg = f"Preprocessing failed: {e}"
            logger.error(error_msg)
            raise PreprocessingError(error_msg, "main_pipeline") from e
    
    def _map_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Map column names to standardized format
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with mapped column names
        """
        logger.debug("Mapping column names to standard format")
        
        column_mapping = self.config.get_column_mapping()
        
        # Only rename columns that exist in the DataFrame
        existing_mapping = {
            old_col: new_col for old_col, new_col in column_mapping.items()
            if old_col in df.columns
        }
        
        df_mapped = df.rename(columns=existing_mapping)
        
        # Log column mapping results
        if existing_mapping:
            logger.debug(f"Mapped columns: {existing_mapping}")
        
        missing_columns = [
            col for col in [self.config.text_column, self.config.subject_id_column]
            if col not in df.columns
        ]
        
        if missing_columns:
            raise PreprocessingError(
                f"Required columns missing: {missing_columns}",
                "column_mapping",
                len(missing_columns)
            )
        
        return df_mapped
    
    def _clean_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and convert data types
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with cleaned data types
        """
        logger.debug("Cleaning data types and handling missing values")
        df = df.copy()
        
        try:
            # Clean text column
            if 'text' in df.columns:
                df['text'] = df['text'].fillna('')
                df['text'] = df['text'].astype(str)
                df['text'] = df['text'].apply(self._clean_text)
                self.processing_stats['data_cleaning_applied'] += len(df)
            
            # Clean subject_id
            if 'subject_id' in df.columns:
                df['subject_id'] = pd.to_numeric(df['subject_id'], errors='coerce')
                self.processing_stats['type_conversions'] += 1
            
            # Clean hadm_id
            if 'hadm_id' in df.columns:
                df['hadm_id'] = pd.to_numeric(df['hadm_id'], errors='coerce')
            
            # Clean categorical fields
            for col in ['category', 'description']:
                if col in df.columns:
                    df[col] = df[col].fillna('')
                    df[col] = df[col].astype(str)
                    df[col] = df[col].str.strip()
            
            # Clean datetime fields
            datetime_columns = ['chartdate', 'charttime', 'storetime']
            for col in datetime_columns:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Clean numeric fields
            if 'cgid' in df.columns:
                df['cgid'] = pd.to_numeric(df['cgid'], errors='coerce')
            
            # Clean boolean fields
            if 'iserror' in df.columns:
                df['iserror'] = df['iserror'].fillna(False)
                df['iserror'] = df['iserror'].astype(bool)
            
            return df
            
        except Exception as e:
            raise PreprocessingError(
                f"Data type cleaning failed: {e}",
                "data_type_cleaning"
            ) from e
    
    def _apply_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply configured filtering rules
        
        Args:
            df: Input DataFrame
            
        Returns:
            Filtered DataFrame
        """
        logger.debug("Applying filtering rules")
        original_len = len(df)
        
        # Remove rows with invalid subject_id
        df = df.dropna(subset=['subject_id'])
        logger.debug(f"Removed {original_len - len(df)} rows with missing subject_id")
        
        # Text length filters
        if 'text' in df.columns:
            pre_filter_len = len(df)
            df = df[df['text'].str.len() >= self.config.min_text_length]
            df = df[df['text'].str.len() <= self.config.max_text_length]
            logger.debug(f"Removed {pre_filter_len - len(df)} rows due to text length filters")
        
        # Category filter
        if self.config.allowed_categories and 'category' in df.columns:
            pre_filter_len = len(df)
            df = df[df['category'].isin(self.config.allowed_categories)]
            logger.debug(f"Removed {pre_filter_len - len(df)} rows due to category filter")
        
        # Error notes filter
        if self.config.exclude_error_notes and 'iserror' in df.columns:
            pre_filter_len = len(df)
            df = df[df['iserror'] != True]
            logger.debug(f"Removed {pre_filter_len - len(df)} error notes")
        
        logger.info(f"Filtering: {original_len} → {len(df)} rows ({len(df)/original_len:.1%} retained)")
        return df
    
    def _remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove duplicate notes based on text content
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with duplicates removed
        """
        if 'text' not in df.columns:
            return df
        
        logger.debug("Removing duplicate notes")
        original_len = len(df)
        
        # Create text hash for duplicate detection
        df['text_hash'] = df['text'].apply(
            lambda x: pd.util.hash_pandas_object(pd.Series([x]))[0] if x else None
        )
        
        # Remove duplicates based on text hash
        df = df.drop_duplicates(subset=['text_hash'], keep='first')
        df = df.drop(columns=['text_hash'])
        
        duplicates_removed = original_len - len(df)
        self.processing_stats['duplicates_removed'] = duplicates_removed
        
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate notes")
        
        return df
    
    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply sampling if configured
        
        Args:
            df: Input DataFrame
            
        Returns:
            Sampled DataFrame
        """
        if not self.config.sample_size or len(df) <= self.config.sample_size:
            return df
        
        logger.info(f"Sampling {self.config.sample_size} notes from {len(df)} available")
        
        # Use stratified sampling by category if possible
        if 'category' in df.columns and df['category'].nunique() > 1:
            sampled_df = self._stratified_sample(df)
        else:
            sampled_df = df.sample(
                n=self.config.sample_size,
                random_state=self.config.random_seed
            )
        
        return sampled_df
    
    def _stratified_sample(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform stratified sampling by category
        
        Args:
            df: Input DataFrame
            
        Returns:
            Stratified sample DataFrame
        """
        category_counts = df['category'].value_counts()
        total_samples = min(self.config.sample_size, len(df))
        
        sampled_dfs = []
        
        for category, count in category_counts.items():
            # Calculate proportional sample size for this category
            proportion = count / len(df)
            category_sample_size = max(1, int(total_samples * proportion))
            
            category_df = df[df['category'] == category]
            if len(category_df) <= category_sample_size:
                sampled_dfs.append(category_df)
            else:
                sample = category_df.sample(
                    n=category_sample_size,
                    random_state=self.config.random_seed
                )
                sampled_dfs.append(sample)
        
        return pd.concat(sampled_dfs, ignore_index=True)
    
    def _convert_to_notes(self, df: pd.DataFrame) -> List[ClinicalNote]:
        """
        Convert DataFrame rows to ClinicalNote objects
        
        Args:
            df: Input DataFrame
            
        Returns:
            List of ClinicalNote objects
        """
        logger.debug(f"Converting {len(df)} rows to ClinicalNote objects")
        notes = []
        conversion_errors = 0
        
        for idx, row in df.iterrows():
            try:
                note = ClinicalNote(
                    subject_id=int(row['subject_id']) if pd.notna(row['subject_id']) else 0,
                    hadm_id=int(row['hadm_id']) if pd.notna(row.get('hadm_id')) else None,
                    text=str(row.get('text', '')),
                    category=str(row.get('category', '')),
                    description=str(row.get('description', '')),
                    chartdate=row.get('chartdate') if pd.notna(row.get('chartdate')) else None,
                    charttime=row.get('charttime') if pd.notna(row.get('charttime')) else None,
                    storetime=row.get('storetime') if pd.notna(row.get('storetime')) else None,
                    cgid=int(row['cgid']) if pd.notna(row.get('cgid')) else None,
                    iserror=bool(row.get('iserror', False))
                )
                notes.append(note)
                
            except Exception as e:
                conversion_errors += 1
                logger.warning(f"Failed to create note from row {idx}: {e}")
                continue
        
        if conversion_errors > 0:
            logger.warning(f"Failed to convert {conversion_errors} rows to ClinicalNote objects")
        
        logger.info(f"Successfully converted {len(notes)} notes")
        return notes
    
    def _clean_text(self, text: str) -> str:
        """
        Clean clinical text content
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        if not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Remove control characters but preserve line breaks
        text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n|\r', '\n', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()


class ClinicalTextPreprocessor(BasePreprocessor):
    """
    Specialized preprocessor for clinical text with medical domain knowledge
    
    Extends BasePreprocessor with clinical-specific text cleaning and normalization.
    """
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.medical_abbreviations = self._load_medical_abbreviations()
        self.text_normalization_rules = self._create_normalization_rules()
    
    def _load_medical_abbreviations(self) -> Dict[str, str]:
        """
        Load medical abbreviation mappings
        
        Returns:
            Dictionary mapping abbreviations to full forms
        """
        return {
            # Common medical abbreviations
            'pt': 'patient',
            'hx': 'history',
            'dx': 'diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'sx': 'symptoms',
            'yo': 'year old',
            'y/o': 'year old',
            'f/u': 'follow up',
            'w/': 'with',
            'w/o': 'without',
            'c/o': 'complains of',
            's/p': 'status post',
            'r/o': 'rule out',
            
            # Vital signs
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'temp': 'temperature',
            'o2 sat': 'oxygen saturation',
            
            # Units
            'mg': 'milligrams',
            'ml': 'milliliters',
            'mcg': 'micrograms',
            'kg': 'kilograms',
            'lbs': 'pounds'
        }
    
    def _create_normalization_rules(self) -> List[Tuple[str, str]]:
        """
        Create text normalization rules
        
        Returns:
            List of (pattern, replacement) tuples
        """
        return [
            # Normalize common medical notation
            (r'\b(\d+)\s*yo\b', r'\1 year old'),
            (r'\b(\d+)\s*y/o\b', r'\1 year old'),
            (r'\bpt\b', 'patient'),
            (r'\bhx\b', 'history'),
            
            # Normalize medication dosages
            (r'(\d+)\s*mg\b', r'\1 milligrams'),
            (r'(\d+)\s*ml\b', r'\1 milliliters'),
            
            # Normalize vital signs
            (r'\bbp\s*:?\s*(\d+/\d+)', r'blood pressure \1'),
            (r'\bhr\s*:?\s*(\d+)', r'heart rate \1'),
            
            # Remove excessive punctuation
            (r'[.]{3,}', '...'),
            (r'[-]{3,}', '---'),
            
            # Normalize spacing around punctuation
            (r'\s*([,.;:!?])\s*', r'\1 '),
            (r'\s+', ' ')
        ]
    
    def _clean_text(self, text: str) -> str:
        """
        Enhanced clinical text cleaning
        
        Args:
            text: Raw clinical text
            
        Returns:
            Cleaned and normalized clinical text
        """
        if not isinstance(text, str):
            return ""
        
        # Apply base cleaning
        text = super()._clean_text(text)
        
        # Apply medical-specific normalization
        text = self._normalize_medical_text(text)
        
        # Apply abbreviation expansion (optional)
        if hasattr(self.config, 'expand_abbreviations') and self.config.expand_abbreviations:
            text = self._expand_abbreviations(text)
        
        return text
    
    def _normalize_medical_text(self, text: str) -> str:
        """
        Apply medical-specific text normalization
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Apply normalization rules
        for pattern, replacement in self.text_normalization_rules:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Fix common clinical text issues
        text = self._fix_clinical_formatting(text)
        
        return text
    
    def _fix_clinical_formatting(self, text: str) -> str:
        """
        Fix common clinical text formatting issues
        
        Args:
            text: Input text
            
        Returns:
            Text with fixed formatting
        """
        # Fix section headers
        text = re.sub(r'\n\s*([A-Z][A-Z\s]+):\s*\n', r'\n\n\1:\n', text)
        
        # Fix bullet points
        text = re.sub(r'\n\s*[-*]\s*', r'\n• ', text)
        
        # Fix medication lists
        text = re.sub(r'(\d+\.)\s*([A-Z])', r'\1 \2', text)
        
        return text
    
    def _expand_abbreviations(self, text: str) -> str:
        """
        Expand medical abbreviations
        
        Args:
            text: Input text with abbreviations
            
        Returns:
            Text with expanded abbreviations
        """
        # Use word boundaries to avoid partial matches
        for abbrev, expansion in self.medical_abbreviations.items():
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            text = re.sub(pattern, expansion, text, flags=re.IGNORECASE)
        
        return text


def create_preprocessor(config: DatasetConfig, preprocessor_type: str = "clinical") -> BasePreprocessor:
    """
    Factory function to create appropriate preprocessor
    
    Args:
        config: Dataset configuration
        preprocessor_type: Type of preprocessor ("base", "clinical")
        
    Returns:
        Configured preprocessor instance
        
    Raises:
        ValueError: If preprocessor type is not recognized
    """
    preprocessor_types = {
        "base": BasePreprocessor,
        "clinical": ClinicalTextPreprocessor
    }
    
    if preprocessor_type not in preprocessor_types:
        raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")
    
    return preprocessor_types[preprocessor_type](config)