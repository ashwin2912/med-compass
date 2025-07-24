#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Named Entity Recognition
Rule-based entity extraction for clinical text
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Types of clinical entities to extract"""
    # Demographics
    AGE = "AGE"
    GENDER = "GENDER"
    
    # Clinical Conditions
    CONDITION = "CONDITION"
    SYMPTOM = "SYMPTOM"
    
    # Medications
    MEDICATION = "MEDICATION"
    DOSAGE = "DOSAGE"
    
    # Measurements
    VITAL_SIGN = "VITAL_SIGN"
    LAB_VALUE = "LAB_VALUE"
    
    # Procedures
    PROCEDURE = "PROCEDURE"
    TEST = "TEST"
    
    # Anatomy
    BODY_PART = "BODY_PART"
    
    # Temporal
    DATE = "DATE"
    DURATION = "DURATION"


@dataclass
class Entity:
    """Represents an extracted clinical entity"""
    text: str
    label: EntityType
    start: int
    end: int
    confidence: float = 1.0
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    def to_dict(self) -> Dict:
        """Convert entity to dictionary"""
        return {
            'text': self.text,
            'label': self.label.value,
            'start': self.start,
            'end': self.end,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class BaseExtractor(ABC):
    """Base class for all entity extractors"""
    
    def __init__(self, entity_type: EntityType):
        self.entity_type = entity_type
        self.patterns = self._compile_patterns()
    
    @abstractmethod
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile regex patterns for this entity type"""
        pass
    
    def extract(self, text: str) -> List[Entity]:
        """Extract entities of this type from text"""
        entities = []
        
        for pattern in self.patterns:
            for match in pattern.finditer(text):
                entity = self._create_entity_from_match(match, text)
                if entity:
                    entities.append(entity)
        
        return entities
    
    def _create_entity_from_match(self, match: re.Match, text: str) -> Optional[Entity]:
        """Create an Entity object from a regex match"""
        entity_text = match.group().strip()
        
        # Skip very short matches
        if len(entity_text) < 2:
            return None
        
        # Calculate confidence based on pattern specificity
        confidence = self._calculate_confidence(match, text)
        
        # Extract metadata if available
        metadata = self._extract_metadata(match, text)
        
        return Entity(
            text=entity_text,
            label=self.entity_type,
            start=match.start(),
            end=match.end(),
            confidence=confidence,
            metadata=metadata
        )
    
    def _calculate_confidence(self, match: re.Match, text: str) -> float:
        """Calculate confidence score for the match (override in subclasses)"""
        return 0.8  # Default confidence
    
    def _extract_metadata(self, match: re.Match, text: str) -> Dict:
        """Extract additional metadata from the match (override in subclasses)"""
        return {}


class VitalSignExtractor(BaseExtractor):
    """Extractor for vital signs (BP, HR, RR, O2, temp)"""
    
    def __init__(self):
        super().__init__(EntityType.VITAL_SIGN)
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile patterns for vital signs"""
        return [
            # Blood Pressure: BP 120/80, blood pressure 140/90, 130/85 mmHg
            re.compile(
                r'(?:bp|blood pressure)[\s:]*(\d{2,3}/\d{2,3})(?:\s*mmhg)?|'
                r'(?<![\d/])\d{2,3}/\d{2,3}(?:\s*mmhg)(?!\d)',
                re.IGNORECASE
            ),
            
            # Heart Rate: HR 78, heart rate 85 bpm, pulse 72
            re.compile(
                r'(?:hr|heart rate|pulse)[\s:]*(\d{1,3})(?:\s*bpm)?|'
                r'(?:rate|pulse)[\s:]+(\d{1,3})(?:\s*bpm)',
                re.IGNORECASE
            ),
            
            # Respiratory Rate: RR 18, respiratory rate 20, breathing 16/min
            re.compile(
                r'(?:rr|respiratory rate|breathing)[\s:]*(\d{1,2})(?:\s*/?min)?',
                re.IGNORECASE
            ),
            
            # Oxygen Saturation: O2 sat 96%, SpO2 98%, oxygen 94%
            re.compile(
                r'(?:o2 sat|spo2|oxygen saturation|oxygen)[\s:]*(\d{1,3})%?',
                re.IGNORECASE
            ),
            
            # Temperature: temp 98.6, temperature 101.3 F, 99.2°F
            re.compile(
                r'(?:temp|temperature)[\s:]*(\d{1,3}\.?\d?)(?:\s*[°]?[fc])?|'
                r'(?<![\d.])\d{1,3}\.\d[°]?[fc](?!\d)',
                re.IGNORECASE
            ),
            
            # Weight: weight 70kg, 150 lbs, wt 68 kg
            re.compile(
                r'(?:weight|wt)[\s:]*(\d{1,3}(?:\.\d)?)\s*(?:kg|lbs|pounds)?|'
                r'(?<![\d.])\d{1,3}(?:\.\d)?\s*(?:kg|lbs)(?!\d)',
                re.IGNORECASE
            )
        ]
    
    def _calculate_confidence(self, match: re.Match, text: str) -> float:
        """Calculate confidence for vital sign matches"""
        match_text = match.group().lower()
        
        # Higher confidence for explicit labels
        if any(label in match_text for label in ['bp', 'blood pressure', 'heart rate', 'hr']):
            return 0.95
        elif any(label in match_text for label in ['temp', 'temperature', 'o2 sat', 'spo2']):
            return 0.90
        elif re.search(r'\d{2,3}/\d{2,3}', match_text):  # BP pattern
            return 0.85
        else:
            return 0.75
    
    def _extract_metadata(self, match: re.Match, text: str) -> Dict:
        """Extract vital sign specific metadata"""
        match_text = match.group().lower()
        metadata = {}
        
        # Determine vital sign type
        if 'bp' in match_text or 'blood pressure' in match_text or '/' in match_text:
            metadata['vital_type'] = 'blood_pressure'
            # Extract systolic/diastolic if possible
            bp_match = re.search(r'(\d{2,3})/(\d{2,3})', match_text)
            if bp_match:
                metadata['systolic'] = int(bp_match.group(1))
                metadata['diastolic'] = int(bp_match.group(2))
        
        elif any(term in match_text for term in ['hr', 'heart rate', 'pulse']):
            metadata['vital_type'] = 'heart_rate'
            # Extract BPM value
            hr_match = re.search(r'(\d{1,3})', match_text)
            if hr_match:
                metadata['bpm'] = int(hr_match.group(1))
        
        elif any(term in match_text for term in ['rr', 'respiratory rate', 'breathing']):
            metadata['vital_type'] = 'respiratory_rate'
        
        elif any(term in match_text for term in ['o2', 'spo2', 'oxygen']):
            metadata['vital_type'] = 'oxygen_saturation'
            # Extract percentage
            o2_match = re.search(r'(\d{1,3})', match_text)
            if o2_match:
                metadata['percentage'] = int(o2_match.group(1))
        
        elif any(term in match_text for term in ['temp', 'temperature']):
            metadata['vital_type'] = 'temperature'
            # Extract value and unit
            temp_match = re.search(r'(\d{1,3}\.?\d?)', match_text)
            if temp_match:
                metadata['value'] = float(temp_match.group(1))
                if 'f' in match_text:
                    metadata['unit'] = 'fahrenheit'
                elif 'c' in match_text:
                    metadata['unit'] = 'celsius'
        
        elif any(term in match_text for term in ['weight', 'wt']):
            metadata['vital_type'] = 'weight'
            if 'kg' in match_text:
                metadata['unit'] = 'kg'
            elif any(unit in match_text for unit in ['lbs', 'pounds']):
                metadata['unit'] = 'lbs'
        
        return metadata


class AgeExtractor(BaseExtractor):
    """Extractor for patient ages"""
    
    def __init__(self):
        super().__init__(EntityType.AGE)
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile patterns for ages"""
        return [
            # 65-year-old, 71 year old, 45 yo, 33 y/o
            re.compile(
                r'(\d{1,3})[-\s]*(?:year[s]?[- ]?old|yo|y/o)',
                re.IGNORECASE
            ),
            
            # Age 65, aged 71
            re.compile(
                r'(?:age[d]?)[\s:]+(\d{1,3})',
                re.IGNORECASE
            )
        ]
    
    def _calculate_confidence(self, match: re.Match, text: str) -> float:
        """Calculate confidence for age matches"""
        match_text = match.group().lower()
        
        if 'year' in match_text:
            return 0.95
        elif any(pattern in match_text for pattern in ['yo', 'y/o']):
            return 0.90
        else:
            return 0.85
    
    def _extract_metadata(self, match: re.Match, text: str) -> Dict:
        """Extract age metadata"""
        # Extract the numeric age
        age_match = re.search(r'(\d{1,3})', match.group())
        metadata = {}
        
        if age_match:
            age_value = int(age_match.group(1))
            metadata['age_value'] = age_value
            
            # Categorize age group
            if age_value < 18:
                metadata['age_group'] = 'pediatric'
            elif age_value < 65:
                metadata['age_group'] = 'adult'
            else:
                metadata['age_group'] = 'geriatric'
        
        return metadata


class MedicationExtractor(BaseExtractor):
    """Extractor for medications and dosages"""
    
    def __init__(self):
        self.common_medications = self._load_common_medications()
        super().__init__(EntityType.MEDICATION)
    
    def _load_common_medications(self) -> Set[str]:
        """Load list of common medications"""
        return {
            'aspirin', 'tylenol', 'acetaminophen', 'ibuprofen', 'advil',
            'lisinopril', 'metoprolol', 'atorvastatin', 'simvastatin',
            'metformin', 'insulin', 'prednisone', 'albuterol', 'furosemide',
            'warfarin', 'coumadin', 'heparin', 'morphine', 'oxycodone',
            'hydrocodone', 'tramadol', 'gabapentin', 'sertraline', 'zoloft',
            'lexapro', 'prozac', 'lipitor', 'zestril', 'lopressor',
            'glucophage', 'lasix', 'proventil', 'ventolin', 'deltasone'
        }
    
    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile patterns for medications"""
        # Create pattern for known medications
        med_names = '|'.join(re.escape(med) for med in self.common_medications)
        
        return [
            # Known medication with dosage: aspirin 81mg, lisinopril 10mg daily
            re.compile(
                f'({med_names})\\s+(\\d+(?:\\.\\d+)?)\\s*(mg|mcg|g|ml|units?)(?:\\s+(daily|bid|tid|qid|prn|q\\d+h?))?',
                re.IGNORECASE
            ),
            
            # Generic medication pattern: [word] [number][unit] [frequency]
            re.compile(
                r'([a-z]+(?:ol|in|ide|ine|ate|one))\\s+(\\d+(?:\\.\\d+)?)\\s*(mg|mcg|g|ml|units?)(?:\\s+(daily|bid|tid|qid|prn|q\\d+h?))?',
                re.IGNORECASE
            )
        ]
    
    def _calculate_confidence(self, match: re.Match, text: str) -> float:
        """Calculate confidence for medication matches"""
        match_text = match.group().lower()
        
        # Higher confidence for known medications
        for med in self.common_medications:
            if med in match_text:
                return 0.95
        
        # Lower confidence for generic pattern matches
        return 0.70
    
    def _extract_metadata(self, match: re.Match, text: str) -> Dict:
        """Extract medication metadata"""
        groups = match.groups()
        metadata = {}
        
        if len(groups) >= 3:
            medication_name = groups[0].lower()
            dosage = groups[1]
            unit = groups[2].lower() if groups[2] else None
            frequency = groups[3].lower() if len(groups) > 3 and groups[3] else None
            
            metadata['medication_name'] = medication_name
            metadata['dosage'] = float(dosage) if dosage else None
            metadata['unit'] = unit
            metadata['frequency'] = frequency
            
            # Classify medication type (basic classification)
            if medication_name in ['aspirin', 'ibuprofen', 'acetaminophen']:
                metadata['medication_class'] = 'analgesic'
            elif medication_name in ['lisinopril', 'metoprolol']:
                metadata['medication_class'] = 'cardiovascular'
            elif medication_name in ['metformin', 'insulin']:
                metadata['medication_class'] = 'diabetes'
            elif medication_name in ['prednisone']:
                metadata['medication_class'] = 'steroid'
        
        return metadata


class ClinicalNER:
    """
    Main Clinical Named Entity Recognition class
    
    Coordinates multiple extractors to identify clinical entities in text.
    """
    
    def __init__(self, extractors: Optional[List[BaseExtractor]] = None):
        """
        Initialize clinical NER
        
        Args:
            extractors: List of entity extractors to use (default: all available)
        """
        if extractors is None:
            self.extractors = [
                VitalSignExtractor(),
                AgeExtractor(),
                MedicationExtractor()
            ]
        else:
            self.extractors = extractors
        
        logger.info(f"Clinical NER initialized with {len(self.extractors)} extractors")
    
    def extract_entities(self, text: str) -> List[Entity]:
        """
        Extract all entities from clinical text
        
        Args:
            text: Clinical text to analyze
            
        Returns:
            List of extracted entities
        """
        if not text or not text.strip():
            return []
        
        logger.debug(f"Extracting entities from text of length {len(text)}")
        
        all_entities = []
        
        # Run each extractor
        for extractor in self.extractors:
            try:
                entities = extractor.extract(text)
                all_entities.extend(entities)
                logger.debug(f"{extractor.__class__.__name__}: found {len(entities)} entities")
            
            except Exception as e:
                logger.warning(f"Error in {extractor.__class__.__name__}: {e}")
                continue
        
        # Post-process entities
        processed_entities = self._post_process_entities(all_entities, text)
        
        logger.debug(f"Total entities extracted: {len(processed_entities)}")
        return processed_entities
    
    def _post_process_entities(self, entities: List[Entity], text: str) -> List[Entity]:
        """
        Post-process entities to remove overlaps and improve quality
        
        Args:
            entities: Raw extracted entities
            text: Original text
            
        Returns:
            Cleaned list of entities
        """
        if not entities:
            return []
        
        # Sort by start position
        entities.sort(key=lambda e: e.start)
        
        # Remove overlapping entities (keep higher confidence)
        non_overlapping = []
        
        for entity in entities:
            # Check if this entity overlaps with any existing entity
            overlaps = False
            
            for existing in non_overlapping:
                if self._entities_overlap(entity, existing):
                    # Keep the entity with higher confidence
                    if entity.confidence > existing.confidence:
                        non_overlapping.remove(existing)
                        non_overlapping.append(entity)
                    overlaps = True
                    break
            
            if not overlaps:
                non_overlapping.append(entity)
        
        # Filter out very low confidence entities
        filtered_entities = [e for e in non_overlapping if e.confidence >= 0.5]
        
        # Sort by position again
        filtered_entities.sort(key=lambda e: e.start)
        
        return filtered_entities
    
    def _entities_overlap(self, entity1: Entity, entity2: Entity) -> bool:
        """Check if two entities overlap in text position"""
        return not (entity1.end <= entity2.start or entity2.end <= entity1.start)
    
    def get_entities_by_type(self, entities: List[Entity], entity_type: EntityType) -> List[Entity]:
        """Filter entities by type"""
        return [e for e in entities if e.label == entity_type]
    
    def entities_to_dict(self, entities: List[Entity]) -> List[Dict]:
        """Convert entities to dictionary format"""
        return [entity.to_dict() for entity in entities]


# Convenience function
def extract_clinical_entities(text: str) -> List[Entity]:
    """
    Convenience function for extracting clinical entities
    
    Args:
        text: Clinical text to analyze
        
    Returns:
        List of extracted entities
    """
    ner = ClinicalNER()
    return ner.extract_entities(text)

'''
# Example usage and testing
if __name__ == "__main__":
    # Test with sample clinical text
    sample_text = """
    Patient is a 65-year-old male with history of hypertension and diabetes mellitus 
    who presented with chest pain. Vital signs: BP 140/90, HR 78, RR 18, O2 sat 96%, 
    temp 98.6°F. Troponin peaked at 45.2 ng/mL. Started on aspirin 81mg daily, 
    lisinopril 10mg daily, and metoprolol 25mg twice daily.
    """
    
    # Extract entities
    entities = extract_clinical_entities(sample_text)
    
    print(f"Extracted {len(entities)} entities:")
    print("=" * 50)
    
    for entity in entities:
        print(f"Text: '{entity.text}'")
        print(f"Type: {entity.label.value}")
        print(f"Position: {entity.start}-{entity.end}")
        print(f"Confidence: {entity.confidence:.2f}")
        if entity.metadata:
            print(f"Metadata: {entity.metadata}")
        print("-" * 30)
    
    # Group by entity type
    ner = ClinicalNER()
    vitals = ner.get_entities_by_type(entities, EntityType.VITAL_SIGN)
    medications = ner.get_entities_by_type(entities, EntityType.MEDICATION)
    ages = ner.get_entities_by_type(entities, EntityType.AGE)
    
    print(f"\nSummary:")
    print(f"Vital Signs: {len(vitals)}")
    print(f"Medications: {len(medications)}")  
    print(f"Ages: {len(ages)}")'''