#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Simple Data Loaders
Basic CSV loading for clinical data ingestion
"""

import pandas as pd
import logging
from pathlib import Path
from typing import Optional
from abc import ABC, abstractmethod

from .models import DatasetConfig
from .exceptions import DataLoadError, SchemaValidationError

logger = logging.getLogger(__name__)


class BaseDataLoader(ABC):
    """Base class for data loaders"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
    
    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load data from file"""
        pass
    
    @abstractmethod
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate DataFrame schema"""
        pass


class CSVLoader(BaseDataLoader):
    """Simple CSV loader for clinical data"""
    
    def load(self) -> pd.DataFrame:
        """Load CSV data with basic error handling"""
        logger.info(f"Loading CSV data from: {self.config.file_path}")
        
        if not Path(self.config.file_path).exists():
            raise DataLoadError(f"File not found: {self.config.file_path}")
        
        try:
            # Try standard CSV loading first
            df = pd.read_csv(
                self.config.file_path,
                encoding=self.config.encoding,
                dtype=str  # Read everything as strings initially
            )
            
            logger.info(f"Successfully loaded {len(df)} rows")
            return df
            
        except UnicodeDecodeError:
            # Fallback to latin1 encoding
            logger.warning("UTF-8 failed, trying latin1 encoding")
            try:
                df = pd.read_csv(
                    self.config.file_path,
                    encoding="latin1",
                    dtype=str
                )
                logger.info(f"Successfully loaded {len(df)} rows with latin1 encoding")
                return df
            except Exception as e:
                raise DataLoadError(f"Failed to load CSV with both UTF-8 and latin1: {e}")
        
        except Exception as e:
            raise DataLoadError(f"Failed to load CSV: {e}")
    
    def validate_schema(self, df: pd.DataFrame) -> bool:
        """Validate that required columns exist"""
        required_columns = [
            self.config.text_column,
            self.config.subject_id_column
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            available = list(df.columns)
            raise SchemaValidationError(
                f"Missing required columns: {missing_columns}. Available: {available}",
                missing_columns,
                available
            )
        
        if len(df) == 0:
            logger.warning("DataFrame is empty")
            return False
        
        logger.info("✓ Schema validation passed")
        return True


class DataLoaderFactory:
    """Simple factory for creating data loaders"""
    
    @staticmethod
    def create_loader(config: DatasetConfig) -> BaseDataLoader:
        """Create appropriate data loader (CSV only for now)"""
        
        # For now, only support CSV
        if config.file_type.lower() != "csv":
            logger.warning(f"File type '{config.file_type}' not fully supported yet, treating as CSV")
        
        return CSVLoader(config)