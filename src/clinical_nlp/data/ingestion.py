#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Data Ingestion Orchestrator
Main orchestrator that coordinates data loading, validation, and preprocessing
"""

import logging
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

from .models import (
    ClinicalNote, DatasetConfig, IngestionMetadata, 
    ValidationResult, DataQuality
)
from .loaders import DataLoaderFactory, BaseDataLoader
from .validators import create_validator, BaseValidator
from .preprocessors import create_preprocessor, BasePreprocessor
from .exceptions import (
    DataIngestionError, DataLoadError, ValidationError, 
    PreprocessingError, ConfigurationError
)

logger = logging.getLogger(__name__)


class ClinicalDataIngestion:
    """
    Main orchestrator for clinical data ingestion pipeline
    
    Coordinates the entire data ingestion process from raw files to validated
    ClinicalNote objects, providing comprehensive error handling, logging,
    and metadata tracking.
    """
    
    def __init__(
        self, 
        config: DatasetConfig,
        loader: Optional[BaseDataLoader] = None,
        validator: Optional[BaseValidator] = None,
        preprocessor: Optional[BasePreprocessor] = None
    ):
        """
        Initialize the data ingestion pipeline
        
        Args:
            config: Dataset configuration object
            loader: Optional custom data loader (auto-created if None)
            validator: Optional custom validator (auto-created if None)
            preprocessor: Optional custom preprocessor (auto-created if None)
        """
        self.config = config
        self.start_time = None
        self.end_time = None
        
        # Initialize components (use factory methods if not provided)
        self.loader = loader or self._create_loader()
        self.validator = validator or self._create_validator()
        self.preprocessor = preprocessor or self._create_preprocessor()
        
        # Initialize tracking variables
        self.raw_data = None
        self.processed_notes = None
        self.valid_notes = None
        self.metadata = None
        
        # Validate configuration
        self._validate_configuration()
        
        logger.info("Clinical data ingestion pipeline initialized")
    
    def _create_loader(self) -> BaseDataLoader:
        """Create appropriate data loader based on configuration"""
        try:
            return DataLoaderFactory.create_loader(self.config)
        except Exception as e:
            raise ConfigurationError(f"Failed to create data loader: {e}") from e
    
    def _create_validator(self) -> BaseValidator:
        """Create appropriate validator based on configuration"""
        try:
            return create_validator(self.config, "composite")
        except Exception as e:
            raise ConfigurationError(f"Failed to create validator: {e}") from e
    
    def _create_preprocessor(self) -> BasePreprocessor:
        """Create appropriate preprocessor based on configuration"""
        try:
            return create_preprocessor(self.config, "clinical")
        except Exception as e:
            raise ConfigurationError(f"Failed to create preprocessor: {e}") from e
    
    def _validate_configuration(self):
        """Validate the configuration before processing"""
        # Check if file exists
        if not Path(self.config.file_path).exists():
            raise ConfigurationError(
                f"Data file not found: {self.config.file_path}",
                "file_path",
                self.config.file_path
            )
        
        # Check file permissions
        if not Path(self.config.file_path).is_file():
            raise ConfigurationError(
                f"Path is not a file: {self.config.file_path}",
                "file_path",
                self.config.file_path
            )
        
        # Validate text length parameters
        if self.config.min_text_length >= self.config.max_text_length:
            raise ConfigurationError(
                "min_text_length must be less than max_text_length",
                "text_length_config"
            )
        
        logger.debug("Configuration validation passed")
    
    def ingest(self) -> Tuple[List[ClinicalNote], IngestionMetadata]:
        """
        Execute the complete data ingestion pipeline
        
        Returns:
            Tuple of (valid_notes, metadata)
            
        Raises:
            DataIngestionError: If any step of the pipeline fails
        """
        logger.info("Starting clinical data ingestion pipeline...")
        logger.debug("Entered ingest function")
        self.start_time = datetime.now()
        
        try:
            # Step 1: Load raw data
            self._load_data()
            
            # Step 2: Validate schema
            self._validate_schema()
            
            # Step 3: Preprocess data
            self._preprocess_data()
            
            # Step 4: Validate clinical content
            self._validate_content()
            
            # Step 5: Generate metadata
            self._generate_metadata()
            
            self.end_time = datetime.now()
            self.metadata.pipeline_end_time = self.end_time
            self.metadata.calculate_duration()
            
            logger.info("✓ Data ingestion pipeline completed successfully")
            logger.info(f"  Duration: {self.metadata.duration_seconds:.2f} seconds")
            logger.info(f"  Valid notes: {len(self.valid_notes)}")
            logger.info(f"  Success rate: {self.metadata.success_rate:.1%}")
            
            return self.valid_notes, self.metadata
            
        except Exception as e:
            self.end_time = datetime.now()
            error_msg = f"Data ingestion pipeline failed: {e}"
            logger.error(error_msg)
            
            # Still try to generate partial metadata for debugging
            try:
                self._generate_partial_metadata(str(e))
            except:
                pass
            
            raise DataIngestionError(error_msg) from e
    
    def _load_data(self):
        """Load raw data using the configured loader"""
        logger.info("Step 1: Loading raw data...")
        
        try:
            self.raw_data = self.loader.load()
            logger.info(f"✓ Loaded {len(self.raw_data)} raw records")
            
        except Exception as e:
            raise DataLoadError(
                f"Failed to load data: {e}",
                self.config.file_path,
                e
            ) from e
    
    def _validate_schema(self):
        """Validate that the loaded data has the expected schema"""
        logger.info("Step 2: Validating data schema...")
        
        try:
            is_valid = self.loader.validate_schema(self.raw_data)
            if not is_valid:
                raise ValidationError("Schema validation failed")
            
            logger.info("✓ Schema validation passed")
            
        except Exception as e:
            raise ValidationError(f"Schema validation failed: {e}") from e
    
    def _preprocess_data(self):
        """Preprocess the raw data into ClinicalNote objects"""
        logger.info("Step 3: Preprocessing data...")
        
        try:
            self.processed_notes = self.preprocessor.preprocess(self.raw_data)
            logger.info(f"✓ Preprocessed {len(self.processed_notes)} notes")
            
        except Exception as e:
            raise PreprocessingError(f"Data preprocessing failed: {e}") from e
    
    def _validate_content(self):
        """Validate the clinical content of the processed notes"""
        logger.info("Step 4: Validating clinical content...")
        
        try:
            validation_result = self.validator.validate(self.processed_notes)
            self.valid_notes = validation_result.valid_notes
            
            if not self.valid_notes:
                raise ValidationError("No valid notes remaining after validation")
            
            logger.info(f"✓ Content validation complete: {len(self.valid_notes)} valid notes")
            
            # Log validation warnings if any
            if validation_result.warning_messages:
                logger.warning(f"Validation warnings: {len(validation_result.warning_messages)}")
                for warning in validation_result.warning_messages[:5]:  # Log first 5 warnings
                    logger.warning(f"  {warning}")
            
        except Exception as e:
            raise ValidationError(f"Content validation failed: {e}") from e
    
    def _generate_metadata(self):
        """Generate comprehensive metadata about the ingestion process"""
        logger.debug("Step 5: Generating metadata...")
        
        try:
            # Create metadata object
            self.metadata = IngestionMetadata(
                pipeline_start_time=self.start_time,
                pipeline_end_time=self.end_time,
                config_hash=self._calculate_config_hash(),
                raw_rows_loaded=len(self.raw_data) if self.raw_data is not None else 0,
                processed_notes_created=len(self.processed_notes) if self.processed_notes else 0,
                valid_notes_final=len(self.valid_notes) if self.valid_notes else 0,
                validation_errors_count=0,  # Will be updated if we have validation results
                success_rate=self._calculate_success_rate()
            )
            
            # Add quality statistics
            if self.valid_notes:
                self.metadata.add_quality_stats(self.valid_notes)
            
            # Add processing statistics from components
            self._add_component_stats()
            
            logger.debug("✓ Metadata generation complete")
            
        except Exception as e:
            logger.warning(f"Failed to generate complete metadata: {e}")
            # Create minimal metadata
            self._generate_minimal_metadata()
    
    def _generate_partial_metadata(self, error_message: str):
        """Generate partial metadata when pipeline fails"""
        self.metadata = IngestionMetadata(
            pipeline_start_time=self.start_time,
            pipeline_end_time=self.end_time,
            config_hash=self._calculate_config_hash(),
            raw_rows_loaded=len(self.raw_data) if self.raw_data is not None else 0,
            processed_notes_created=len(self.processed_notes) if self.processed_notes else 0,
            valid_notes_final=len(self.valid_notes) if self.valid_notes else 0,
            validation_errors_count=1,
            success_rate=0.0,
            processing_warnings=[f"Pipeline failed: {error_message}"]
        )
        
        if self.end_time:
            self.metadata.calculate_duration()
    
    def _generate_minimal_metadata(self):
        """Generate minimal metadata when full metadata generation fails"""
        self.metadata = IngestionMetadata(
            pipeline_start_time=self.start_time,
            pipeline_end_time=self.end_time or datetime.now(),
            config_hash=self._calculate_config_hash(),
            raw_rows_loaded=len(self.raw_data) if self.raw_data is not None else 0,
            processed_notes_created=len(self.processed_notes) if self.processed_notes else 0,
            valid_notes_final=len(self.valid_notes) if self.valid_notes else 0,
            validation_errors_count=0,
            success_rate=self._calculate_success_rate()
        )
        
        self.metadata.calculate_duration()
    
    def _calculate_config_hash(self) -> str:
        """Calculate a hash of the configuration for tracking"""
        config_str = str(self.config.dict())
        return hashlib.md5(config_str.encode()).hexdigest()[:16]
    
    def _calculate_success_rate(self) -> float:
        """Calculate the overall success rate"""
        if not self.processed_notes:
            return 0.0
        
        valid_count = len(self.valid_notes) if self.valid_notes else 0
        total_count = len(self.processed_notes)
        
        return valid_count / total_count if total_count > 0 else 0.0
    
    def _add_component_stats(self):
        """Add statistics from individual components to metadata"""
        # Add preprocessor statistics
        if hasattr(self.preprocessor, 'processing_stats'):
            stats = self.preprocessor.processing_stats
            self.metadata.duplicate_notes_removed = stats.get('duplicates_removed', 0)
            self.metadata.notes_filtered_by_length = stats.get('rows_filtered', 0)
        
        # Add validator statistics (if available)
        if hasattr(self.validator, 'validation_stats'):
            validator_stats = self.validator.validation_stats
            # Add validator-specific metrics
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the pipeline execution
        
        Returns:
            Dictionary containing pipeline summary information
        """
        if not self.metadata:
            return {"status": "not_executed"}
        
        summary = {
            "status": "completed" if self.valid_notes else "failed",
            "execution_time": self.metadata.duration_seconds,
            "data_flow": {
                "raw_records": self.metadata.raw_rows_loaded,
                "processed_notes": self.metadata.processed_notes_created,
                "valid_notes": self.metadata.valid_notes_final,
                "success_rate": f"{self.metadata.success_rate:.1%}"
            },
            "quality_distribution": self.metadata.quality_distribution,
            "category_distribution": self.metadata.category_distribution,
            "text_statistics": {
                "avg_length": f"{self.metadata.text_stats.get('avg_text_length', 0):.0f} chars",
                "avg_words": f"{self.metadata.text_stats.get('avg_word_count', 0):.0f} words"
            },
            "issues": {
                "validation_errors": self.metadata.validation_errors_count,
                "warnings": len(self.metadata.processing_warnings),
                "duplicates_removed": self.metadata.duplicate_notes_removed
            }
        }
        
        return summary
    
    def save_results(
        self, 
        output_dir: str = "outputs", 
        save_notes: bool = True,
        save_metadata: bool = True,
        save_summary: bool = True
    ):
        """
        Save pipeline results to files
        
        Args:
            output_dir: Directory to save results
            save_notes: Whether to save valid notes
            save_metadata: Whether to save metadata
            save_summary: Whether to save pipeline summary
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if save_notes and self.valid_notes:
            # Save notes as JSON
            notes_data = [note.to_dict() for note in self.valid_notes]
            notes_file = output_path / f"clinical_notes_{timestamp}.json"
            
            import json
            with open(notes_file, 'w') as f:
                json.dump(notes_data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(self.valid_notes)} notes to {notes_file}")
        
        if save_metadata and self.metadata:
            # Save metadata as JSON
            metadata_file = output_path / f"ingestion_metadata_{timestamp}.json"
            
            import json
            with open(metadata_file, 'w') as f:
                json.dump(self.metadata.dict(), f, indent=2, default=str)
            
            logger.info(f"Saved metadata to {metadata_file}")
        
        if save_summary:
            # Save summary as JSON
            summary = self.get_pipeline_summary()
            summary_file = output_path / f"pipeline_summary_{timestamp}.json"
            
            import json
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Saved pipeline summary to {summary_file}")


# Convenience functions for common use cases

def ingest_mimic_data(
    file_path: str,
    sample_size: Optional[int] = None,
    categories: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Tuple[List[ClinicalNote], IngestionMetadata]:
    """
    Convenience function to ingest MIMIC-III/IV data
    
    Args:
        file_path: Path to MIMIC NOTEEVENTS.csv file
        sample_size: Optional sample size for testing
        categories: Optional list of note categories to include
        output_dir: Optional directory to save results
        
    Returns:
        Tuple of (valid_notes, metadata)
    """
    from .models import create_mimic_config
    
    # Create MIMIC-optimized configuration
    config = create_mimic_config(file_path, sample_size, categories)
    
    # Create and run ingestion pipeline
    ingestion = ClinicalDataIngestion(config)
    notes, metadata = ingestion.ingest()
    
    # Save results if output directory specified
    if output_dir:
        ingestion.save_results(output_dir)
    
    return notes, metadata


def ingest_custom_data(
    config: DatasetConfig,
    loader_type: Optional[str] = None,
    validator_type: str = "composite",
    preprocessor_type: str = "clinical",
    output_dir: Optional[str] = None
) -> Tuple[List[ClinicalNote], IngestionMetadata]:
    """
    Convenience function to ingest custom clinical data
    
    Args:
        config: Dataset configuration
        loader_type: Optional loader type override
        validator_type: Type of validator to use
        preprocessor_type: Type of preprocessor to use
        output_dir: Optional directory to save results
        
    Returns:
        Tuple of (valid_notes, metadata)
    """
    # Create components
    if loader_type:
        # Override config file type if specified
        config.file_type = loader_type
    
    loader = DataLoaderFactory.create_loader(config)
    validator = create_validator(config, validator_type)
    preprocessor = create_preprocessor(config, preprocessor_type)
    
    # Create and run ingestion pipeline
    ingestion = ClinicalDataIngestion(config, loader, validator, preprocessor)
    notes, metadata = ingestion.ingest()
    
    # Save results if output directory specified
    if output_dir:
        ingestion.save_results(output_dir)
    
    return notes, metadata


def create_demo_pipeline(output_dir: str = "demo_output") -> Tuple[List[ClinicalNote], IngestionMetadata]:
    """
    Create and run a demonstration pipeline with sample data
    
    Args:
        output_dir: Directory to save demo results
        
    Returns:
        Tuple of (valid_notes, metadata)
    """
    import pandas as pd
    import tempfile
    import os
    
    # Create sample clinical notes
    sample_data = pd.DataFrame([
        {
            'SUBJECT_ID': 10001,
            'HADM_ID': 20001,
            'TEXT': """Patient is a 65-year-old male with history of hypertension and diabetes mellitus type 2 who presented with chest pain and shortness of breath. EKG showed ST elevation in leads II, III, aVF suggesting inferior STEMI. Patient was taken emergently to cardiac catheterization where 100% occlusion of RCA was found and successfully treated with drug-eluting stent. Post-procedure troponin peaked at 45. Patient had uncomplicated recovery and was discharged on dual antiplatelet therapy, ACE inhibitor, and beta-blocker. Follow-up with cardiology in 1 week.""",
            'CATEGORY': 'Discharge summary',
            'DESCRIPTION': 'Discharge Summary'
        },
        {
            'SUBJECT_ID': 10002,
            'HADM_ID': 20002,
            'TEXT': """45-year-old female with asthma exacerbation. Patient reports increased wheezing, cough, and dyspnea over past 3 days. Peak flow decreased from baseline 400 to 200. Physical exam notable for expiratory wheeze throughout lung fields. Started on albuterol nebulizers q4h, prednisolone 40mg daily. Chest X-ray clear. Patient responding well to treatment with improved peak flow to 350.""",
            'CATEGORY': 'Physician',
            'DESCRIPTION': 'Physician Progress Note'
        },
        {
            'SUBJECT_ID': 10003,
            'HADM_ID': 20003,
            'TEXT': """Patient admitted with acute kidney injury secondary to dehydration. Creatinine on admission 3.2, baseline 1.1. Started on IV fluid resuscitation. Patient has good urine output, appears clinically improved. Labs this morning show creatinine down to 2.1. Continue current management. Patient ambulating without difficulty.""",
            'CATEGORY': 'Nursing',
            'DESCRIPTION': 'Nursing Progress Note'
        },
        {
            'SUBJECT_ID': 10004,
            'HADM_ID': 20004,
            'TEXT': "Short note",  # This should fail validation
            'CATEGORY': 'Progress Note',
            'DESCRIPTION': 'Brief Note'
        }
    ])
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        temp_file = f.name
    
    try:
        # Create demo configuration
        from .models import create_demo_config
        config = create_demo_config(temp_file)
        
        # Run ingestion pipeline
        notes, metadata = ingest_custom_data(config, output_dir=output_dir)
        
        logger.info(f"Demo pipeline completed: {len(notes)} valid notes generated")
        return notes, metadata
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.unlink(temp_file)


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo pipeline
    print("Running Clinical NLP Data Ingestion Demo...")
    print("=" * 60)
    
    try:
        notes, metadata = create_demo_pipeline()
        
        print(f"\n✓ Demo completed successfully!")
        print(f"  Valid notes: {len(notes)}")
        print(f"  Success rate: {metadata.success_rate:.1%}")
        print(f"  Processing time: {metadata.duration_seconds:.2f}s")
        
        print(f"\nQuality Distribution:")
        for quality, count in metadata.quality_distribution.items():
            print(f"  {quality}: {count}")
        
        print(f"\nFirst note preview:")
        if notes:
            note = notes[0]
            print(f"  Subject ID: {note.subject_id}")
            print(f"  Category: {note.category}")
            print(f"  Quality: {note.data_quality.value}")
            print(f"  Text: {note.text[:150]}...")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        raise