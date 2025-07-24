#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Data Validators
Comprehensive validation for clinical notes with medical domain knowledge
"""

import re
import logging
from typing import List, Dict, Tuple, Set, Pattern, Optional, Any
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from datetime import datetime

from .models import ClinicalNote, DatasetConfig, ValidationResult, DataQuality
from .exceptions import ValidationError

logger = logging.getLogger(__name__)


class BaseValidator(ABC):
    """
    Abstract base class for all data validators
    
    Defines the interface that all concrete validators must implement.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize validator with configuration
        
        Args:
            config: Dataset configuration object
        """
        self.config = config
        self.validation_stats = {
            'total_processed': 0,
            'passed': 0,
            'failed': 0,
            'warnings': 0
        }
    
    @abstractmethod
    def validate(self, notes: List[ClinicalNote]) -> ValidationResult:
        """
        Validate a list of clinical notes
        
        Args:
            notes: List of clinical notes to validate
            
        Returns:
            ValidationResult containing valid notes and error information
        """
        pass
    
    @abstractmethod
    def validate_single_note(self, note: ClinicalNote, index: int) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single clinical note
        
        Args:
            note: Clinical note to validate
            index: Index of the note for error reporting
            
        Returns:
            Tuple of (is_valid, error_messages, warning_messages)
        """
        pass


class ClinicalContentValidator(BaseValidator):
    """
    Validator for clinical content with medical domain knowledge
    
    Validates that notes contain appropriate medical content, terminology,
    and structure expected in clinical documentation.
    """
    
    def __init__(self, config: DatasetConfig):
        """
        Initialize clinical content validator
        
        Args:
            config: Dataset configuration object
        """
        super().__init__(config)
        self.medical_patterns = self._compile_medical_patterns()
        self.medical_terms = self._load_medical_terminology()
        self.validation_rules = self._initialize_validation_rules()
    
    def _compile_medical_patterns(self) -> Dict[str, Pattern]:
        """
        Compile regex patterns for medical content validation
        
        Returns:
            Dictionary of compiled regex patterns
        """
        patterns = {
            # Basic medical terms
            'medical_terms': re.compile(
                r'\b(patient|pt|diagnosis|treatment|medication|symptom|condition|'
                r'history|hx|exam|examination|procedure|therapy|clinical|medical|'
                r'hospital|doctor|physician|nurse|provider|care|health|disease|'
                r'disorder|syndrome|injury|pain|fever|infection|inflammation|'
                r'blood|pressure|heart|lung|chest|abdomen|head|brain|kidney|liver)\b',
                re.IGNORECASE
            ),
            
            # Clinical structure indicators
            'structured_format': re.compile(
                r'\b(history|examination|assessment|plan|impression|diagnosis|'
                r'chief complaint|present illness|review of systems|physical exam|'
                r'labs|imaging|medications|allergies|discharge|admission|'
                r'subjective|objective|assessment|plan|soap)\b',
                re.IGNORECASE
            ),
            
            # Medication patterns
            'medications': re.compile(
                r'\b\w+\s*\d+\s*(mg|ml|mcg|g|units?|tablets?)\b|'
                r'\b(aspirin|tylenol|ibuprofen|morphine|insulin|metformin|lisinopril|'
                r'atorvastatin|amlodipine|omeprazole|albuterol|prednisone|warfarin|'
                r'metoprolol|furosemide|gabapentin|sertraline|levothyroxine)\b',
                re.IGNORECASE
            ),
            
            # Vital signs and measurements
            'vitals': re.compile(
                r'\b(bp|blood pressure|hr|heart rate|temp|temperature|rr|'
                r'respiratory rate|o2|oxygen|spo2|pulse|weight|height|bmi)\s*:?\s*'
                r'\d+(/\d+)?(\.\d+)?\s*(mmhg|bpm|f|c|kg|lbs|cm|inches?)?\b',
                re.IGNORECASE
            ),
            
            # Lab values
            'lab_values': re.compile(
                r'\b(glucose|creatinine|bun|sodium|potassium|chloride|co2|'
                r'hemoglobin|hematocrit|wbc|platelets|inr|pt|ptt|troponin|'
                r'ck|alt|ast|bilirubin|albumin|protein)\s*:?\s*\d+(\.\d+)?\s*'
                r'(mg/dl|mmol/l|g/dl|k/ul|sec)?\b',
                re.IGNORECASE
            ),
            
            # Procedures and tests
            'procedures': re.compile(
                r'\b(ct|mri|x-ray|xray|ultrasound|echo|ekg|ecg|colonoscopy|'
                r'endoscopy|biopsy|surgery|catheterization|intubation|'
                r'ventilation|dialysis|transfusion|injection)\b',
                re.IGNORECASE
            ),
            
            # Clinical specialties
            'specialties': re.compile(
                r'\b(cardiology|neurology|oncology|gastroenterology|pulmonology|'
                r'nephrology|endocrinology|psychiatry|surgery|emergency|'
                r'radiology|pathology|anesthesiology|dermatology|orthopedic)\b',
                re.IGNORECASE
            )
        }
        
        return patterns
    
    def _load_medical_terminology(self) -> Dict[str, Set[str]]:
        """
        Load medical terminology dictionaries
        
        Returns:
            Dictionary of medical term categories
        """
        return {
            'common_conditions': {
                'hypertension', 'diabetes', 'asthma', 'copd', 'pneumonia',
                'myocardial infarction', 'stroke', 'sepsis', 'cancer',
                'heart failure', 'atrial fibrillation', 'depression',
                'anxiety', 'obesity', 'arthritis', 'osteoporosis'
            },
            
            'symptoms': {
                'chest pain', 'shortness of breath', 'dyspnea', 'nausea',
                'vomiting', 'diarrhea', 'constipation', 'headache',
                'dizziness', 'fatigue', 'weakness', 'confusion',
                'syncope', 'palpitations', 'cough', 'fever'
            },
            
            'body_systems': {
                'cardiovascular', 'respiratory', 'gastrointestinal',
                'neurological', 'musculoskeletal', 'genitourinary',
                'endocrine', 'hematologic', 'dermatologic', 'psychiatric'
            },
            
            'anatomical_terms': {
                'heart', 'lung', 'brain', 'liver', 'kidney', 'stomach',
                'intestine', 'bladder', 'bone', 'muscle', 'skin',
                'blood vessel', 'artery', 'vein', 'nerve'
            }
        }
    
    def _initialize_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """
        Initialize validation rules with scoring weights
        
        Returns:
            Dictionary of validation rules and their parameters
        """
        return {
            'minimum_medical_terms': {
                'threshold': 3,
                'weight': 0.3,
                'description': 'Minimum number of medical terms required'
            },
            'structured_content': {
                'threshold': 1,
                'weight': 0.2,
                'description': 'Presence of clinical structure indicators'
            },
            'medical_measurements': {
                'threshold': 1,
                'weight': 0.2,
                'description': 'Presence of vital signs, labs, or medications'
            },
            'clinical_context': {
                'threshold': 2,
                'weight': 0.3,
                'description': 'Overall clinical context and coherence'
            }
        }
    
    def validate(self, notes: List[ClinicalNote]) -> ValidationResult:
        """
        Validate a list of clinical notes
        
        Args:
            notes: List of clinical notes to validate
            
        Returns:
            ValidationResult with valid notes and validation statistics
        """
        logger.info(f"Validating {len(notes)} clinical notes for medical content...")
        
        result = ValidationResult()
        error_categories = defaultdict(int)
        
        for i, note in enumerate(notes):
            is_valid, errors, warnings = self.validate_single_note(note, i)
            
            if is_valid:
                result.add_valid_note(note)
            else:
                for error in errors:
                    # Extract error category from error message
                    category = self._extract_error_category(error)
                    result.add_error(error, category)
                    error_categories[category] += 1
            
            # Add warnings regardless of validation result
            for warning in warnings:
                result.add_warning(warning)
        
        result.finalize()
        
        # Log validation summary
        logger.info(f"Medical content validation complete:")
        logger.info(f"  Valid notes: {result.notes_passed}")
        logger.info(f"  Failed notes: {result.notes_failed}")
        logger.info(f"  Success rate: {result.notes_passed/len(notes):.1%}")
        
        if error_categories:
            logger.info("  Error categories:")
            for category, count in error_categories.items():
                logger.info(f"    {category}: {count}")
        
        return result
    
    def validate_single_note(self, note: ClinicalNote, index: int) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a single clinical note for medical content
        
        Args:
            note: Clinical note to validate
            index: Index for error reporting
            
        Returns:
            Tuple of (is_valid, error_messages, warning_messages)
        """
        errors = []
        warnings = []
        
        # Skip validation if medical content validation is disabled
        if not self.config.validate_medical_content:
            return True, errors, warnings
        
        text = note.text.lower()
        
        # 1. Basic content validation
        basic_errors = self._validate_basic_content(note, index)
        errors.extend(basic_errors)
        
        # 2. Medical terminology validation
        medical_score = self._calculate_medical_score(text)
        if medical_score < 0.5:  # Threshold for medical content
            errors.append(f"Note {index}: Insufficient medical content (score: {medical_score:.2f})")
        elif medical_score < 0.7:
            warnings.append(f"Note {index}: Low medical content score ({medical_score:.2f})")
        
        # 3. Clinical structure validation
        structure_warnings = self._validate_clinical_structure(text, index)
        warnings.extend(structure_warnings)
        
        # 4. Category-specific validation
        category_errors = self._validate_by_category(note, index)
        errors.extend(category_errors)
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    def _validate_basic_content(self, note: ClinicalNote, index: int) -> List[str]:
        """Validate basic content requirements"""
        errors = []
        
        # Text length validation
        if note.text_length < self.config.min_text_length:
            errors.append(f"Note {index}: Text too short ({note.text_length} chars)")
        
        if note.text_length > self.config.max_text_length:
            errors.append(f"Note {index}: Text too long ({note.text_length} chars)")
        
        # Category validation
        if (self.config.allowed_categories and 
            note.category not in self.config.allowed_categories):
            errors.append(f"Note {index}: Invalid category '{note.category}'")
        
        # Error note filtering
        if self.config.exclude_error_notes and note.iserror:
            errors.append(f"Note {index}: Marked as error note")
        
        # Empty or whitespace-only text
        if not note.text or not note.text.strip():
            errors.append(f"Note {index}: Empty or whitespace-only text")
        
        return errors
    
    def _calculate_medical_score(self, text: str) -> float:
        """
        Calculate medical content score for text
        
        Args:
            text: Text to analyze (should be lowercase)
            
        Returns:
            Score between 0 and 1 indicating medical content quality
        """
        score = 0.0
        
        # Count matches for each pattern type
        pattern_scores = {}
        for pattern_name, pattern in self.medical_patterns.items():
            matches = len(pattern.findall(text))
            
            # Normalize scores based on pattern importance and frequency
            if pattern_name == 'medical_terms':
                pattern_scores[pattern_name] = min(matches / 5, 1.0) * 0.3
            elif pattern_name == 'structured_format':
                pattern_scores[pattern_name] = min(matches / 3, 1.0) * 0.2
            elif pattern_name == 'medications':
                pattern_scores[pattern_name] = min(matches / 2, 1.0) * 0.15
            elif pattern_name == 'vitals':
                pattern_scores[pattern_name] = min(matches / 3, 1.0) * 0.15
            elif pattern_name == 'lab_values':
                pattern_scores[pattern_name] = min(matches / 2, 1.0) * 0.1
            elif pattern_name == 'procedures':
                pattern_scores[pattern_name] = min(matches / 2, 1.0) * 0.05
            elif pattern_name == 'specialties':
                pattern_scores[pattern_name] = min(matches / 1, 1.0) * 0.05
        
        # Sum weighted scores
        score = sum(pattern_scores.values())
        
        # Bonus for medical terminology diversity
        unique_medical_terms = set()
        for term_category, terms in self.medical_terms.items():
            for term in terms:
                if term in text:
                    unique_medical_terms.add(term)
        
        diversity_bonus = min(len(unique_medical_terms) / 10, 0.2)
        score += diversity_bonus
        
        return min(score, 1.0)
    
    def _validate_clinical_structure(self, text: str, index: int) -> List[str]:
        """Validate clinical note structure"""
        warnings = []
        
        # Check for common clinical note sections
        expected_sections = [
            'history', 'examination', 'assessment', 'plan',
            'chief complaint', 'physical exam'
        ]
        
        found_sections = []
        for section in expected_sections:
            if section in text:
                found_sections.append(section)
        
        if len(found_sections) == 0:
            warnings.append(f"Note {index}: No recognizable clinical sections found")
        elif len(found_sections) == 1:
            warnings.append(f"Note {index}: Limited clinical structure (only '{found_sections[0]}')")
        
        return warnings
    
    def _validate_by_category(self, note: ClinicalNote, index: int) -> List[str]:
        """Validate note based on its category"""
        errors = []
        category = note.category.lower()
        text = note.text.lower()
        
        # Discharge summary specific validation
        if 'discharge' in category:
            required_elements = ['diagnosis', 'medication', 'follow']
            missing_elements = [elem for elem in required_elements if elem not in text]
            if len(missing_elements) > 1:
                errors.append(f"Note {index}: Discharge summary missing key elements: {missing_elements}")
        
        # Progress note specific validation
        elif 'progress' in category:
            if not any(word in text for word in ['assessment', 'plan', 'progress', 'status']):
                errors.append(f"Note {index}: Progress note lacks progress indicators")
        
        # Nursing note specific validation
        elif 'nursing' in category:
            if not any(word in text for word in ['patient', 'care', 'vital', 'status', 'comfort']):
                errors.append(f"Note {index}: Nursing note lacks typical nursing content")
        
        return errors
    
    def _extract_error_category(self, error_message: str) -> str:
        """Extract error category from error message"""
        if 'too short' in error_message or 'too long' in error_message:
            return 'length_validation'
        elif 'medical content' in error_message:
            return 'medical_content'
        elif 'category' in error_message:
            return 'category_validation'
        elif 'error note' in error_message:
            return 'error_flag'
        elif 'empty' in error_message:
            return 'empty_content'
        else:
            return 'general'


class DataQualityValidator(BaseValidator):
    """
    Validator focused on data quality metrics
    
    Validates data completeness, consistency, and identifies potential data issues.
    """
    
    def __init__(self, config: DatasetConfig):
        super().__init__(config)
        self.quality_thresholds = {
            DataQuality.HIGH: {'min_length': 200, 'min_words': 30},
            DataQuality.MEDIUM: {'min_length': 50, 'min_words': 10},
            DataQuality.LOW: {'min_length': 10, 'min_words': 3}
        }
    
    def validate(self, notes: List[ClinicalNote]) -> ValidationResult:
        """
        Validate data quality across all notes
        
        Args:
            notes: List of clinical notes to validate
            
        Returns:
            ValidationResult with quality assessment
        """
        logger.info(f"Performing data quality validation on {len(notes)} notes...")
        
        result = ValidationResult()
        quality_stats = Counter()
        
        for i, note in enumerate(notes):
            is_valid, errors, warnings = self.validate_single_note(note, i)
            
            # Track quality distribution
            quality_stats[note.data_quality] += 1
            
            if is_valid:
                result.add_valid_note(note)
            else:
                for error in errors:
                    result.add_error(error, 'data_quality')
            
            for warning in warnings:
                result.add_warning(warning)
        
        result.finalize()
        
        # Log quality distribution
        logger.info("Data quality distribution:")
        for quality, count in quality_stats.items():
            percentage = count / len(notes) * 100
            logger.info(f"  {quality.value}: {count} ({percentage:.1f}%)")
        
        return result
    
    def validate_single_note(self, note: ClinicalNote, index: int) -> Tuple[bool, List[str], List[str]]:
        """
        Validate single note for data quality issues
        
        Args:
            note: Clinical note to validate
            index: Index for error reporting
            
        Returns:
            Tuple of (is_valid, error_messages, warning_messages)
        """
        errors = []
        warnings = []
        
        # 1. Completeness validation
        completeness_errors = self._validate_completeness(note, index)
        errors.extend(completeness_errors)
        
        # 2. Consistency validation
        consistency_warnings = self._validate_consistency(note, index)
        warnings.extend(consistency_warnings)
        
        # 3. Data quality assessment
        if note.data_quality == DataQuality.INVALID:
            errors.append(f"Note {index}: Invalid data quality")
        elif note.data_quality == DataQuality.LOW:
            warnings.append(f"Note {index}: Low data quality")
        
        is_valid = len(errors) == 0
        return is_valid, errors, warnings
    
    def _validate_completeness(self, note: ClinicalNote, index: int) -> List[str]:
        """Validate data completeness"""
        errors = []
        
        # Required fields
        if not note.subject_id:
            errors.append(f"Note {index}: Missing subject_id")
        
        if not note.text or not note.text.strip():
            errors.append(f"Note {index}: Missing or empty text content")
        
        # Warn about missing optional but important fields
        if not note.category:
            # This is a warning, not an error
            pass
        
        return errors
    
    def _validate_consistency(self, note: ClinicalNote, index: int) -> List[str]:
        """Validate data consistency"""
        warnings = []
        
        # Check for obvious inconsistencies
        if note.hadm_id and note.hadm_id <= 0:
            warnings.append(f"Note {index}: Invalid hadm_id value: {note.hadm_id}")
        
        if note.subject_id and note.subject_id <= 0:
            warnings.append(f"Note {index}: Invalid subject_id value: {note.subject_id}")
        
        # Date consistency checks
        if note.chartdate and note.charttime:
            if note.chartdate.date() != note.charttime.date():
                warnings.append(f"Note {index}: Inconsistent chart date and time")
        
        return warnings


class CompositeValidator(BaseValidator):
    """
    Composite validator that runs multiple validators in sequence
    
    Combines results from multiple validation strategies to provide
    comprehensive validation with detailed error categorization.
    """
    
    def __init__(self, config: DatasetConfig, validators: Optional[List[BaseValidator]] = None):
        """
        Initialize composite validator
        
        Args:
            config: Dataset configuration
            validators: List of validators to run (default: creates standard set)
        """
        super().__init__(config)
        
        if validators is None:
            self.validators = [
                DataQualityValidator(config),
                ClinicalContentValidator(config)
            ]
        else:
            self.validators = validators
    
    def validate(self, notes: List[ClinicalNote]) -> ValidationResult:
        """
        Run all validators and combine results
        
        Args:
            notes: List of clinical notes to validate
            
        Returns:
            Combined ValidationResult from all validators
        """
        logger.info(f"Running composite validation with {len(self.validators)} validators...")
        
        # Start with all notes as potentially valid
        current_notes = notes.copy()
        combined_result = ValidationResult()
        all_errors = []
        all_warnings = []
        
        # Run each validator in sequence
        for i, validator in enumerate(self.validators):
            validator_name = validator.__class__.__name__
            logger.info(f"Running validator {i+1}/{len(self.validators)}: {validator_name}")
            
            result = validator.validate(current_notes)
            
            # Collect errors and warnings
            all_errors.extend(result.error_messages)
            all_warnings.extend(result.warning_messages)
            
            # Update current notes to only include those that passed
            current_notes = result.valid_notes
            
            logger.info(f"  {validator_name}: {len(result.valid_notes)}/{len(current_notes)} passed")
        
        # Build final result
        combined_result.valid_notes = current_notes
        combined_result.error_messages = all_errors
        combined_result.warning_messages = all_warnings
        combined_result.finalize()
        
        logger.info(f"Composite validation complete: {len(combined_result.valid_notes)} notes passed all validators")
        
        return combined_result
    
    def validate_single_note(self, note: ClinicalNote, index: int) -> Tuple[bool, List[str], List[str]]:
        """
        Validate single note with all validators
        
        Args:
            note: Clinical note to validate
            index: Index for error reporting
            
        Returns:
            Combined validation result
        """
        all_errors = []
        all_warnings = []
        is_valid = True
        
        for validator in self.validators:
            valid, errors, warnings = validator.validate_single_note(note, index)
            
            if not valid:
                is_valid = False
            
            all_errors.extend(errors)
            all_warnings.extend(warnings)
        
        return is_valid, all_errors, all_warnings


# Factory function for creating validators

def create_validator(config: DatasetConfig, validator_type: str = "composite") -> BaseValidator:
    """
    Factory function to create appropriate validator
    
    Args:
        config: Dataset configuration
        validator_type: Type of validator ("composite", "clinical", "quality")
        
    Returns:
        Configured validator instance
        
    Raises:
        ValueError: If validator type is not recognized
    """
    validator_types = {
        "composite": CompositeValidator,
        "clinical": ClinicalContentValidator,
        "quality": DataQualityValidator
    }
    
    if validator_type not in validator_types:
        raise ValueError(f"Unknown validator type: {validator_type}")
    
    return validator_types[validator_type](config)