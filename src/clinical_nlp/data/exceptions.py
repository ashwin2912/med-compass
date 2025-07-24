#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Custom Exceptions
Custom exception classes for data ingestion and processing
"""


class ClinicalNLPError(Exception):
    """Base exception class for all clinical NLP pipeline errors"""
    
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class DataIngestionError(ClinicalNLPError):
    """Base exception for data ingestion layer errors"""
    pass


class DataLoadError(DataIngestionError):
    """Exception raised when data loading fails"""
    
    def __init__(self, message: str, file_path: str = None, original_error: Exception = None):
        details = {}
        if file_path:
            details['file_path'] = file_path
        if original_error:
            details['original_error'] = str(original_error)
            details['error_type'] = type(original_error).__name__
        
        super().__init__(message, details)
        self.file_path = file_path
        self.original_error = original_error


class SchemaValidationError(DataIngestionError):
    """Exception raised when data schema validation fails"""
    
    def __init__(self, message: str, missing_columns: list = None, available_columns: list = None):
        details = {}
        if missing_columns:
            details['missing_columns'] = missing_columns
        if available_columns:
            details['available_columns'] = available_columns
        
        super().__init__(message, details)
        self.missing_columns = missing_columns or []
        self.available_columns = available_columns or []


class ValidationError(DataIngestionError):
    """Exception raised when clinical note validation fails"""
    
    def __init__(self, message: str, failed_notes: int = None, error_categories: dict = None):
        details = {}
        if failed_notes is not None:
            details['failed_notes'] = failed_notes
        if error_categories:
            details['error_categories'] = error_categories
        
        super().__init__(message, details)
        self.failed_notes = failed_notes
        self.error_categories = error_categories or {}


class PreprocessingError(DataIngestionError):
    """Exception raised during data preprocessing"""
    
    def __init__(self, message: str, processing_step: str = None, affected_rows: int = None):
        details = {}
        if processing_step:
            details['processing_step'] = processing_step
        if affected_rows is not None:
            details['affected_rows'] = affected_rows
        
        super().__init__(message, details)
        self.processing_step = processing_step
        self.affected_rows = affected_rows


class ConfigurationError(ClinicalNLPError):
    """Exception raised when configuration is invalid"""
    
    def __init__(self, message: str, config_field: str = None, config_value: str = None):
        details = {}
        if config_field:
            details['config_field'] = config_field
        if config_value:
            details['config_value'] = config_value
        
        super().__init__(message, details)
        self.config_field = config_field
        self.config_value = config_value