#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Data Ingestion Layer
Modular data loading, validation, and preprocessing for clinical notes
"""

from .models import (
    ClinicalNote,
    DatasetConfig,
    IngestionMetadata,
    ValidationResult,
    NoteType,
    DataQuality,
    FileType,
    create_mimic_config,
    create_demo_config
)

from .loaders import (
    BaseDataLoader,
    CSVLoader,
    DataLoaderFactory
)

from .validators import (
    BaseValidator,
    ClinicalContentValidator,
    DataQualityValidator,
    CompositeValidator,
    create_validator
)

from .preprocessors import (
    BasePreprocessor,
    ClinicalTextPreprocessor,
    create_preprocessor
)

from .ingestion import (
    ClinicalDataIngestion,
    ingest_mimic_data,
    ingest_custom_data,
    create_demo_pipeline
)

from .exceptions import (
    ClinicalNLPError,
    DataIngestionError,
    DataLoadError,
    SchemaValidationError,
    ValidationError,
    PreprocessingError,
    ConfigurationError
)

# Version information
__version__ = "0.1.0"
__author__ = "Clinical NLP Team"

# Main public API
__all__ = [
    # Core models
    "ClinicalNote",
    "DatasetConfig", 
    "IngestionMetadata",
    "ValidationResult",
    
    # Enums
    "NoteType",
    "DataQuality",
    "FileType",
    
    # Configuration factories
    "create_mimic_config",
    "create_demo_config",
    
    # Data loaders
    "BaseDataLoader",
    "CSVLoader",
    "JSONLoader", 
    "ExcelLoader",
    "ParquetLoader",
    "DataLoaderFactory",
    
    # Validators
    "BaseValidator",
    "ClinicalContentValidator",
    "DataQualityValidator", 
    "CompositeValidator",
    "create_validator",
    
    # Preprocessors
    "BasePreprocessor",
    "ClinicalTextPreprocessor",
    "create_preprocessor",
    
    # Main ingestion orchestrator
    "ClinicalDataIngestion",
    
    # Convenience functions
    "ingest_mimic_data",
    "ingest_custom_data", 
    "create_demo_pipeline",
    
    # Exceptions
    "ClinicalNLPError",
    "DataIngestionError",
    "DataLoadError",
    "SchemaValidationError",
    "ValidationError", 
    "PreprocessingError",
    "ConfigurationError"
]

# Package metadata
__package_info__ = {
    "name": "clinical-nlp-data",
    "version": __version__,
    "description": "Clinical NLP Pipeline - Data Ingestion Layer",
    "author": __author__,
    "python_requires": ">=3.8",
    "dependencies": [
        "pandas>=1.5.0",
        "numpy>=1.23.0",
        "pydantic>=1.10.0",
        "pathlib",
        "datetime",
        "typing"
    ]
}

def get_version():
    """Get package version"""
    return __version__

def get_package_info():
    """Get comprehensive package information"""
    return __package_info__