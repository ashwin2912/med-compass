#!/usr/bin/env python3
"""
Clinical NLP Pipeline - Text Preprocessing
Clean and prepare clinical text for NLP analysis
"""

import re
import logging
import nltk
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from nltk.stem import WordNetLemmatizer
from spellchecker import SpellChecker
import string


logger = logging.getLogger(__name__)


class SectionType(str, Enum):
    """Clinical note section types"""
    HISTORY = "history"
    PHYSICAL_EXAM = "physical_exam"
    ASSESSMENT = "assessment"
    PLAN = "plan"
    LABS = "labs"
    MEDICATIONS = "medications"
    DISCHARGE = "discharge"
    OTHER = "other"


@dataclass
class ProcessedText:
    """Container for processed clinical text"""
    original_text: str
    clean_text: str
    sentences: List[str]
    sections: Dict[str, str]
    abbreviations_expanded: Dict[str, str]
    phi_removed_count: int
    metadata: Dict[str, Any]


class ClinicalTextPreprocessor:
    """
    Preprocessor for clinical text with medical domain knowledge
    
    Handles clinical-specific text cleaning, normalization, and structure extraction.
    """
    
    def __init__(self):
        """Initialize the clinical text preprocessor"""
        try:
            nltk.data.find('corpora/wordnet')
        except:
            nltk.download('wordnet')
        self.clinical_abbreviations = self._load_clinical_abbreviations()
        self.section_patterns = self._compile_section_patterns()
        self.phi_patterns = self._compile_phi_patterns()
        self.cleaning_patterns = self._compile_cleaning_patterns()
        self.lemmatizer = WordNetLemmatizer()
        self.spell = SpellChecker()

        logger.info("Clinical text preprocessor initialized")
        
    def _load_clinical_abbreviations(self) -> Dict[str, str]:
        """Load clinical abbreviation mappings"""
        return {
            # Patient/Demographics
            'pt': 'patient',
            'pts': 'patients',
            'yo': 'year old',
            'y/o': 'year old',
            'm': 'male',
            'f': 'female',
            
            # History/Examination
            'hx': 'history',
            'h/o': 'history of',
            'fhx': 'family history',
            'pmh': 'past medical history',
            'psh': 'past surgical history',
            'shx': 'social history',
            'ros': 'review of systems',
            'pe': 'physical examination',
            'heent': 'head eyes ears nose throat',
            
            # Clinical Terms
            'dx': 'diagnosis',
            'ddx': 'differential diagnosis',
            'tx': 'treatment',
            'rx': 'prescription',
            'sx': 'symptoms',
            's/p': 'status post',
            'r/o': 'rule out',
            'w/': 'with',
            'w/o': 'without',
            'c/o': 'complains of',
            'p/w': 'presented with',
            'f/u': 'follow up',
            'fu': 'follow up',
            
            # Medical Conditions
            'htn': 'hypertension',
            'dm': 'diabetes mellitus',
            't2dm': 'type 2 diabetes mellitus',
            'cad': 'coronary artery disease',
            'chf': 'congestive heart failure',
            'copd': 'chronic obstructive pulmonary disease',
            'afib': 'atrial fibrillation',
            'mi': 'myocardial infarction',
            'stemi': 'ST elevation myocardial infarction',
            'nstemi': 'non-ST elevation myocardial infarction',
            'ckd': 'chronic kidney disease',
            'esrd': 'end stage renal disease',
            'uri': 'upper respiratory infection',
            'uti': 'urinary tract infection',
            
            # Vital Signs
            'bp': 'blood pressure',
            'hr': 'heart rate',
            'rr': 'respiratory rate',
            'o2': 'oxygen',
            'spo2': 'oxygen saturation',
            'temp': 'temperature',
            
            # Procedures
            'ekg': 'electrocardiogram',
            'ecg': 'electrocardiogram',
            'cxr': 'chest x-ray',
            'ct': 'computed tomography',
            'mri': 'magnetic resonance imaging',
            'pci': 'percutaneous coronary intervention',
            'cabg': 'coronary artery bypass graft',
            
            # Units (common ones)
            'mg': 'milligrams',
            'mcg': 'micrograms',
            'ml': 'milliliters',
            'kg': 'kilograms',
            'lbs': 'pounds',
            'mmhg': 'millimeters of mercury',
            'bpm': 'beats per minute'
        }
    
    def _compile_section_patterns(self) -> Dict[SectionType, re.Pattern]:
        """Compile regex patterns for clinical section detection"""
        patterns = {
            SectionType.HISTORY: re.compile(
                r'(?:^|\n)\s*(?:history|hpi|history of present illness|chief complaint|cc)[\s:]*\n?',
                re.IGNORECASE | re.MULTILINE
            ),
            SectionType.PHYSICAL_EXAM: re.compile(
                r'(?:^|\n)\s*(?:physical exam|examination|pe|vital signs|vitals)[\s:]*\n?',
                re.IGNORECASE | re.MULTILINE
            ),
            SectionType.ASSESSMENT: re.compile(
                r'(?:^|\n)\s*(?:assessment|impression|diagnosis|a&p|assessment and plan)[\s:]*\n?',
                re.IGNORECASE | re.MULTILINE
            ),
            SectionType.PLAN: re.compile(
                r'(?:^|\n)\s*(?:plan|recommendations|disposition|discharge plans?)[\s:]*\n?',
                re.IGNORECASE | re.MULTILINE
            ),
            SectionType.LABS: re.compile(
                r'(?:^|\n)\s*(?:labs|laboratory|lab results|lab values)[\s:]*\n?',
                re.IGNORECASE | re.MULTILINE
            ),
            SectionType.MEDICATIONS: re.compile(
                r'(?:^|\n)\s*(?:medications|meds|discharge medications|home medications)[\s:]*\n?',
                re.IGNORECASE | re.MULTILINE
            ),
            SectionType.DISCHARGE: re.compile(
                r'(?:^|\n)\s*(?:discharge|discharge summary|discharge instructions)[\s:]*\n?',
                re.IGNORECASE | re.MULTILINE
            )
        }
        return patterns
    
    def _compile_phi_patterns(self) -> List[re.Pattern]:
        """Compile patterns to detect and remove PHI placeholders"""
        return [
            # MIMIC-style PHI placeholders
            re.compile(r'\[\*\*[^\]]+\*\*\]'),
            # Date placeholders
            re.compile(r'\[\*\*\d{4}-\d{1,2}-\d{1,2}\*\*\]'),
            # Name placeholders
            re.compile(r'\[\*\*(?:First Name|Last Name|Name|Doctor|Hospital)\s*[^\]]*\*\*\]'),
            # Other common PHI patterns
            re.compile(r'\[\*\*[A-Z][A-Za-z\s]+\*\*\]'),
            # Phone numbers
            re.compile(r'\b(?:\+?1\s*(?:[.-]\s*)?)?(?:\(?\d{3}\)?|\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b'),
            #Email address
            re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            #SSN
            re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            #Dates
            re.compile(r'\b(?:\d{1,2}[/-])?(?:\d{1,2}[/-])?\d{2,4}\b'),
            #Time
            re.compile(r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM|am|pm)?\b'),
            #Addresses
            re.compile(r'\b\d{1,5}\s+\w+\s+(Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Lane|Ln|Drive|Dr)\b'),
            re.compile(r'\b\d{5}(?:-\d{4})?\b'),  # US ZIP codes
            #ID
            re.compile(r'\bRoom\s?\d+\b', re.IGNORECASE),
            re.compile(r'\bID\s?:?\s?\d+\b', re.IGNORECASE)
        ]
    
    def _compile_cleaning_patterns(self) -> List[Tuple[re.Pattern, str]]:
        """Compile text cleaning patterns"""
        return [
            # Multiple spaces to single space
            (re.compile(r'\s+'), ' '),
            # Multiple newlines to double newline
            (re.compile(r'\n{3,}'), '\n\n'),
            # Remove trailing/leading whitespace from lines
            (re.compile(r'^\s+|\s+$', re.MULTILINE), ''),
            # Fix punctuation spacing
            (re.compile(r'\s*([,.;:!?])\s*'), r'\1 '),
            # Remove excessive punctuation
            (re.compile(r'[.]{3,}'), '...'),
            (re.compile(r'[-]{3,}'), '---'),
        ]
    
    def preprocess(self, text: str) -> ProcessedText:
        """
        Main preprocessing pipeline
        
        Args:
            text: Raw clinical note text
            
        Returns:
            ProcessedText object with cleaned text and metadata
        """
        if not text or not text.strip():
            return self._create_empty_result(text)
        
        logger.debug(f"Preprocessing text of length {len(text)}")
        
        # Step 1: Remove PHI placeholders
        clean_text, phi_count = self._remove_phi_placeholders(text)

        clean_text = self._normalize_case(clean_text)

        clean_text = self._normalize_special_chars(clean_text)

        
        # Step 2: Expand clinical abbreviations
        clean_text, expanded_abbrev = self._expand_abbreviations(clean_text)

        clean_text = self._spell_correct(clean_text)

        clean_text = self._lemmatize(clean_text)


        
        # Step 3: Apply general text cleaning
        clean_text = self._apply_cleaning_patterns(clean_text)
        
        # Step 4: Detect clinical sections
        sections = self._detect_sections(clean_text)
        
        # Step 5: Split into sentences
        sentences = self._split_sentences(clean_text)
        
        # Step 6: Generate metadata
        metadata = self._generate_metadata(text, clean_text, sections, sentences)
        
        result = ProcessedText(
            original_text=text,
            clean_text=clean_text,
            sentences=sentences,
            sections=sections,
            abbreviations_expanded=expanded_abbrev,
            phi_removed_count=phi_count,
            metadata=metadata
        )
        
        logger.debug(f"Preprocessing complete: {len(sentences)} sentences, {len(sections)} sections")
        return result
    
    def _remove_phi_placeholders(self, text: str) -> Tuple[str, int]:
        """Remove PHI placeholders from text"""
        phi_count = 0
        clean_text = text
        
        for pattern in self.phi_patterns:
            matches = pattern.findall(clean_text)
            phi_count += len(matches)
            clean_text = pattern.sub('[REMOVED]', clean_text)
        
        # Clean up multiple [REMOVED] tags
        clean_text = re.sub(r'\[REMOVED\]\s*\[REMOVED\]', '[REMOVED]', clean_text)
        clean_text = re.sub(r'\[REMOVED\]', '', clean_text)
        
        return clean_text, phi_count
    
    def _normalize_case(self, text:str) -> str:
        """Convert all text to lowercase"""
        return text.lower()
    
    def _normalize_special_chars(self, text: str) -> str:
        """Remove or normalize special characters"""
        return re.sub(rf"[\{string.punctuation.replace('/', '').replace('-', '')}]", "", text)
    
    def _lemmatize(self, text: str) -> str:
        """Lemmatize words to base form"""
        return ' '.join(self.lemmatizer.lemmatize(word) for word in text.split())
    
    def _spell_correct(self, text: str) -> str:
        """Correct spelling mistakes"""
        corrected = []
        for word in text.split():
            if word not in self.spell and word.isalpha():
                corrected.append(self.spell.correction(word) or word)
            else:
                corrected.append(word)
        return ' '.join(corrected)
    
    def _expand_abbreviations(self, text: str) -> Tuple[str, Dict[str, str]]:
        """Expand clinical abbreviations"""
        expanded_abbrev = {}
        clean_text = text
        
        # Sort abbreviations by length (longest first) to avoid partial matches
        sorted_abbrevs = sorted(self.clinical_abbreviations.items(), 
                               key=lambda x: len(x[0]), reverse=True)
        
        for abbrev, expansion in sorted_abbrevs:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            
            if re.search(pattern, clean_text, re.IGNORECASE):
                clean_text = re.sub(pattern, expansion, clean_text, flags=re.IGNORECASE)
                expanded_abbrev[abbrev] = expansion
        
        return clean_text, expanded_abbrev
    
    def _apply_cleaning_patterns(self, text: str) -> str:
        """Apply general text cleaning patterns"""
        clean_text = text
        
        for pattern, replacement in self.cleaning_patterns:
            clean_text = pattern.sub(replacement, clean_text)
        
        return clean_text.strip()
    
    def _detect_sections(self, text: str) -> Dict[str, str]:
        """Detect and extract clinical note sections"""
        sections = {}
        
        # Find all section boundaries
        section_positions = []
        for section_type, pattern in self.section_patterns.items():
            for match in pattern.finditer(text):
                section_positions.append((match.start(), section_type, match.group()))
        
        # Sort by position
        section_positions.sort(key=lambda x: x[0])
        
        # Extract text between sections
        for i, (start_pos, section_type, header) in enumerate(section_positions):
            # Find end position (start of next section or end of text)
            if i + 1 < len(section_positions):
                end_pos = section_positions[i + 1][0]
            else:
                end_pos = len(text)
            
            # Extract section content
            section_text = text[start_pos:end_pos]
            
            # Remove the header and clean up
            section_text = section_text[len(header):].strip()
            
            if section_text:  # Only add non-empty sections
                sections[section_type.value] = section_text[:500]  # Limit length
        
        return sections
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (clinical-aware)"""
        # Simple sentence splitting for now
        # Note: Clinical text can be tricky due to abbreviations and formatting
        
        # Split on periods, but be careful with abbreviations and numbers
        sentences = []
        
        # First, protect common abbreviations that end with periods
        protected_text = text
        abbreviation_placeholders = {}
        
        # Protect medical abbreviations that might end sentences incorrectly
        medical_abbrevs_with_period = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'vs.', 'etc.', 'i.e.', 'e.g.']
        
        for i, abbrev in enumerate(medical_abbrevs_with_period):
            placeholder = f"__ABBREV_{i}__"
            abbreviation_placeholders[placeholder] = abbrev
            protected_text = protected_text.replace(abbrev, placeholder)
        
        # Split on sentence endings
        potential_sentences = re.split(r'[.!?]+\s+', protected_text)
        
        # Restore abbreviations and clean up
        for sentence in potential_sentences:
            # Restore abbreviations
            for placeholder, abbrev in abbreviation_placeholders.items():
                sentence = sentence.replace(placeholder, abbrev)
            
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Filter out very short fragments
                sentences.append(sentence)
        
        return sentences
    
    def _generate_metadata(self, original: str, clean: str, sections: Dict, sentences: List) -> Dict[str, Any]:
        """Generate preprocessing metadata"""
        return {
            'original_length': len(original),
            'clean_length': len(clean),
            'compression_ratio': len(clean) / len(original) if original else 0,
            'sentence_count': len(sentences),
            'section_count': len(sections),
            'avg_sentence_length': sum(len(s) for s in sentences) / len(sentences) if sentences else 0,
            'sections_detected': list(sections.keys()),
        }
    
    def _create_empty_result(self, text: str) -> ProcessedText:
        """Create empty result for invalid input"""
        return ProcessedText(
            original_text=text or "",
            clean_text="",
            sentences=[],
            sections={},
            abbreviations_expanded={},
            phi_removed_count=0,
            metadata={'original_length': len(text) if text else 0, 'clean_length': 0}
        )


# Convenience function
def preprocess_clinical_text(text: str) -> ProcessedText:
    """
    Convenience function for preprocessing clinical text
    
    Args:
        text: Raw clinical note text
        
    Returns:
        ProcessedText object with cleaned text and metadata
    """
    preprocessor = ClinicalTextPreprocessor()
    return preprocessor.preprocess(text)
