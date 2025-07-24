#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Data Models & Schemas
Pydantic models and data classes for clinical note processing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum
import hashlib
import numpy as np
from pydantic import BaseModel, validator, Field


class NoteType(str, Enum):
    """Enumeration of clinical note types"""
    DISCHARGE_SUMMARY = "Discharge summary"
    PHYSICIAN_NOTE = "Physician"
    NURSING_NOTE = "Nursing"
    PROGRESS_NOTE = "Progress Note"
    CONSULT_NOTE = "Consult"
    RADIOLOGY = "Radiology"
    ECG = "ECG"
    ECHO = "Echo"
    OTHER = "Other"


class DataQuality(str, Enum):
    """Data quality levels for clinical notes"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INVALID = "invalid"


class FileType(str, Enum):
    """Supported file types for data ingestion"""
    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    EXCEL = "excel"


@dataclass
class ClinicalNote:
    """
    Data class representing a single clinical note with metadata
    
    Attributes:
        subject_id: Patient identifier
        hadm_id: Hospital admission identifier
        text: Clinical note text content
        category: Note category (e.g., Discharge summary, Progress Note)
        description: Detailed note description
        chartdate: Date the note was charted
        charttime: Time the note was charted
        storetime: Time the note was stored in system
        cgid: Caregiver identifier
        iserror: Whether the note is marked as an error
    """
    subject_id: int
    hadm_id: Optional[int] = None
    text: str = ""
    category: str = ""
    description: str = ""
    chartdate: Optional[datetime] = None
    charttime: Optional[datetime] = None
    storetime: Optional[datetime] = None
    cgid: Optional[int] = None
    iserror: Optional[bool] = None
    
    # Derived fields (calculated automatically)
    text_length: int = field(init=False)
    word_count: int = field(init=False)
    sentence_count: int = field(init=False)
    data_quality: DataQuality = field(init=False)
    text_hash: str = field(init=False)
    
    def __post_init__(self):
        """Calculate derived fields after initialization"""
        self.text_length = len(self.text) if self.text else 0
        self.word_count = len(self.text.split()) if self.text else 0
        self.sentence_count = len([s for s in self.text.split('.') if s.strip()]) if self.text else 0
        self.data_quality = self._assess_quality()
        self.text_hash = hashlib.md5(str(self.text).encode()).hexdigest()[:16]
    
    def _assess_quality(self) -> DataQuality:
        """
        Assess the quality of the clinical note based on content metrics
        
        Returns:
            DataQuality: Quality assessment (HIGH, MEDIUM, LOW, INVALID)
        """
        if not self.text or self.text_length < 10:
            return DataQuality.INVALID
        elif self.text_length < 50 or self.word_count < 10:
            return DataQuality.LOW
        elif self.text_length < 200 or self.word_count < 30:
            return DataQuality.MEDIUM
        else:
            return DataQuality.HIGH
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert clinical note to dictionary"""
        return {
            'subject_id': self.subject_id,
            'hadm_id': self.hadm_id,
            'text': self.text,
            'category': self.category,
            'description': self.description,
            'chartdate': self.chartdate.isoformat() if self.chartdate else None,
            'charttime': self.charttime.isoformat() if self.charttime else None,
            'storetime': self.storetime.isoformat() if self.storetime else None,
            'cgid': self.cgid,
            'iserror': self.iserror,
            'text_length': self.text_length,
            'word_count': self.word_count,
            'sentence_count': self.sentence_count,
            'data_quality': self.data_quality.value,
            'text_hash': self.text_hash
        }


class DatasetConfig(BaseModel):
    """
    Configuration model for dataset loading and processing
    
    This class defines all parameters needed to load and process clinical datasets,
    with validation and default values following MIMIC-III conventions.
    """
    
    # File Configuration
    file_path: str = Field(..., description="Path to the dataset file")
    file_type: FileType = Field(default=FileType.CSV, description="Type of file to load")
    encoding: str = Field(default="utf-8", description="File encoding")
    
    # Column Mapping (MIMIC-III defaults)
    text_column: str = Field(default="TEXT", description="Name of text content column")
    subject_id_column: str = Field(default="SUBJECT_ID", description="Name of subject ID column")
    hadm_id_column: str = Field(default="HADM_ID", description="Name of hospital admission ID column")
    category_column: str = Field(default="CATEGORY", description="Name of note category column")
    description_column: str = Field(default="DESCRIPTION", description="Name of note description column")
    chartdate_column: Optional[str] = Field(default="CHARTDATE", description="Name of chart date column")
    charttime_column: Optional[str] = Field(default="CHARTTIME", description="Name of chart time column")
    storetime_column: Optional[str] = Field(default="STORETIME", description="Name of store time column")
    cgid_column: Optional[str] = Field(default="CGID", description="Name of caregiver ID column")
    iserror_column: Optional[str] = Field(default="ISERROR", description="Name of error flag column")
    
    # Filtering Options
    min_text_length: int = Field(default=10, ge=1, description="Minimum text length in characters")
    max_text_length: int = Field(default=50000, ge=100, description="Maximum text length in characters")
    allowed_categories: Optional[List[str]] = Field(
        default=None, 
        description="List of allowed note categories (None = all categories)"
    )
    exclude_error_notes: bool = Field(default=True, description="Whether to exclude notes marked as errors")
    
    # Sampling Options
    sample_size: Optional[int] = Field(
        default=None, 
        ge=1, 
        description="Maximum number of notes to load (None = load all)"
    )
    random_seed: int = Field(default=42, description="Random seed for reproducible sampling")
    
    # Advanced Options
    deduplicate: bool = Field(default=True, description="Remove duplicate notes based on text hash")
    validate_medical_content: bool = Field(
        default=True, 
        description="Validate that notes contain medical content"
    )
    
    @validator('file_path')
    def validate_file_path(cls, v):
        """Validate that file path is not empty"""
        if not v or not v.strip():
            raise ValueError("file_path cannot be empty")
        return v.strip()
    
    @validator('max_text_length')
    def validate_text_length_range(cls, v, values):
        """Validate that max_text_length > min_text_length"""
        if 'min_text_length' in values and v <= values['min_text_length']:
            raise ValueError("max_text_length must be greater than min_text_length")
        return v
    
    @validator('allowed_categories')
    def validate_categories(cls, v):
        """Validate and normalize category names"""
        if v is not None:
            # Remove empty strings and strip whitespace
            v = [cat.strip() for cat in v if cat and cat.strip()]
            if not v:  # If list becomes empty after cleaning
                return None
        return v
    
    def get_column_mapping(self) -> Dict[str, str]:
        """
        Get column mapping dictionary for renaming DataFrame columns
        
        Returns:
            Dict mapping original column names to standardized names
        """
        mapping = {
            self.text_column: 'text',
            self.subject_id_column: 'subject_id',
            self.hadm_id_column: 'hadm_id',
            self.category_column: 'category',
            self.description_column: 'description',
        }
        
        # Add optional columns if specified
        optional_mappings = {
            self.chartdate_column: 'chartdate',
            self.charttime_column: 'charttime',
            self.storetime_column: 'storetime',
            self.cgid_column: 'cgid',
            self.iserror_column: 'iserror'
        }
        
        for original, standard in optional_mappings.items():
            if original:  # Only add if column name is specified
                mapping[original] = standard
        
        return mapping


class IngestionMetadata(BaseModel):
    """
    Metadata model for data ingestion process tracking
    
    Contains comprehensive information about the ingestion process,
    including performance metrics, data quality statistics, and error tracking.
    """
    
    # Pipeline Information
    pipeline_start_time: datetime
    pipeline_end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    config_hash: str
    
    # Data Statistics
    raw_rows_loaded: int
    processed_notes_created: int
    valid_notes_final: int
    validation_errors_count: int
    success_rate: float
    
    # Quality Distribution
    quality_distribution: Dict[str, int] = Field(default_factory=dict)
    category_distribution: Dict[str, int] = Field(default_factory=dict)
    
    # Text Statistics
    text_stats: Dict[str, float] = Field(default_factory=dict)
    
    # Error Tracking
    validation_errors: List[str] = Field(default_factory=list)
    processing_warnings: List[str] = Field(default_factory=list)
    
    # Data Quality Metrics
    duplicate_notes_removed: int = 0
    notes_filtered_by_length: int = 0
    notes_filtered_by_category: int = 0
    notes_filtered_by_medical_content: int = 0
    
    @validator('success_rate')
    def validate_success_rate(cls, v):
        """Ensure success rate is between 0 and 1"""
        return max(0.0, min(1.0, v))
    
    def calculate_duration(self):
        """Calculate and set duration if end time is available"""
        if self.pipeline_end_time:
            self.duration_seconds = (
                self.pipeline_end_time - self.pipeline_start_time
            ).total_seconds()
    
    def add_quality_stats(self, notes: List[ClinicalNote]):
        """Calculate and add quality statistics from notes"""
        if not notes:
            return
        
        # Quality distribution
        quality_counts = {}
        category_counts = {}
        
        for note in notes:
            # Quality distribution
            quality = note.data_quality.value
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
            
            # Category distribution
            category = note.category or "Unknown"
            category_counts[category] = category_counts.get(category, 0) + 1
        
        self.quality_distribution = quality_counts
        self.category_distribution = category_counts
        
        # Text statistics
        text_lengths = [note.text_length for note in notes]
        word_counts = [note.word_count for note in notes]
        sentence_counts = [note.sentence_count for note in notes]
        
        self.text_stats = {
            'avg_text_length': float(np.mean(text_lengths)),
            'avg_word_count': float(np.mean(word_counts)),
            'avg_sentence_count': float(np.mean(sentence_counts)),
            'min_text_length': float(np.min(text_lengths)),
            'max_text_length': float(np.max(text_lengths)),
            'median_text_length': float(np.median(text_lengths)),
            'std_text_length': float(np.std(text_lengths))
        }


class ValidationResult:
    """
    Result class for data validation operations (Regular Python class, not Pydantic)
    
    Contains validation outcomes, error messages, and quality metrics.
    """
    
    def __init__(self):
        self.is_valid = False
        self.valid_notes: List[ClinicalNote] = []
        self.error_messages: List[str] = []
        self.warning_messages: List[str] = []
        
        # Validation Statistics
        self.total_notes_processed = 0
        self.notes_passed = 0
        self.notes_failed = 0
        
        # Error Categories
        self.error_categories: Dict[str, int] = {}
    
    def add_error(self, message: str, category: str = "general"):
        """Add an error message with categorization"""
        self.error_messages.append(message)
        self.error_categories[category] = self.error_categories.get(category, 0) + 1
        self.notes_failed += 1
    
    def add_warning(self, message: str):
        """Add a warning message"""
        self.warning_messages.append(message)
    
    def add_valid_note(self, note: ClinicalNote):
        """Add a valid note to the results"""
        self.valid_notes.append(note)
        self.notes_passed += 1
    
    def finalize(self):
        """Finalize validation results and calculate summary statistics"""
        self.total_notes_processed = self.notes_passed + self.notes_failed
        self.is_valid = self.notes_passed > 0 and len(self.error_messages) == 0


# Factory functions for common configurations

def create_mimic_config(
    file_path: str,
    sample_size: Optional[int] = None,
    categories: Optional[List[str]] = None
) -> DatasetConfig:
    """
    Create a DatasetConfig optimized for MIMIC-III/IV datasets
    
    Args:
        file_path: Path to MIMIC NOTEEVENTS.csv file
        sample_size: Optional sample size for testing
        categories: Optional list of note categories to include
        
    Returns:
        Configured DatasetConfig for MIMIC data
    """
    return DatasetConfig(
        file_path=file_path,
        file_type=FileType.CSV,
        encoding="utf-8",
        min_text_length=50,
        max_text_length=20000,
        allowed_categories=categories or [
            "Discharge summary",
            "Physician",
            "Nursing",
            "Progress Note",
            "Consult"
        ],
        exclude_error_notes=True,
        sample_size=sample_size,
        deduplicate=True,
        validate_medical_content=True
    )


def create_demo_config(file_path: str = "demo_notes.csv") -> DatasetConfig:
    """
    Create a DatasetConfig for demonstration/testing purposes
    
    Args:
        file_path: Path to demo data file
        
    Returns:
        Configured DatasetConfig for demo data
    """
    return DatasetConfig(
        file_path=file_path,
        file_type=FileType.CSV,
        min_text_length=10,
        max_text_length=10000,
        sample_size=100,
        exclude_error_notes=False,
        deduplicate=False,
        validate_medical_content=False
    )