#!/usr/bin/env python3
"""
Test the complete preprocessing + NER pipeline
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ..nlp.preprocessing import preprocess_clinical_text
from ..nlp.entity_extraction import extract_clinical_entities

def test_complete_pipeline():
    """Test preprocessing + NER on sample clinical text"""
    
    # Sample MIMIC-style clinical note
    sample_note = """
    ADMISSION DIAGNOSIS: Acute myocardial infarction

    HISTORY OF PRESENT ILLNESS: 
    The patient is a 71-year-old male with a history of hypertension, diabetes mellitus type 2, 
    and hyperlipidemia who presented to the emergency department with acute onset chest pain.

    PHYSICAL EXAMINATION:
    Vital signs: BP 145/92, HR 78, RR 18, O2 sat 96% on room air, afebrile
    General: Male in moderate distress due to chest pain

    DIAGNOSTIC STUDIES:
    Troponin I peaked at 45.2 ng/mL (normal <0.04)

    DISCHARGE MEDICATIONS:
    1. Aspirin 81mg daily
    2. Clopidogrel 75mg daily 
    3. Atorvastatin 80mg daily
    4. Lisinopril 10mg daily
    5. Metoprolol XL 50mg daily
    """
    
    print("🔄 Testing Complete NLP Pipeline")
    print("=" * 60)
    
    # Step 1: Preprocessing
    print("Step 1: Text Preprocessing...")
    processed = preprocess_clinical_text(sample_note)
    
    print(f"✅ Original length: {len(sample_note)} chars")
    print(f"✅ Clean length: {len(processed.clean_text)} chars")
    print(f"✅ Sections found: {list(processed.sections.keys())}")
    print(f"✅ Abbreviations expanded: {len(processed.abbreviations_expanded)}")
    
    # Step 2: Entity Extraction
    print("\nStep 2: Named Entity Recognition...")
    entities = extract_clinical_entities(processed.clean_text)
    
    print(f"✅ Total entities extracted: {len(entities)}")
    
    # Group by type
    entity_types = {}
    for entity in entities:
        entity_type = entity.label.value
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(entity)
    
    # Show results by type
    print("\n📊 Entities by Type:")
    print("-" * 40)
    
    for entity_type, type_entities in entity_types.items():
        print(f"\n{entity_type} ({len(type_entities)}):")
        for entity in type_entities:
            print(f"  • '{entity.text}' (confidence: {entity.confidence:.2f})")
            if entity.metadata:
                print(f"    Metadata: {entity.metadata}")
    
    # Show clean text with entities highlighted
    print(f"\n📝 Processed Text Preview:")
    print("-" * 40)
    print(processed.clean_text[:300] + "...")
    
    return processed, entities

if __name__ == "__main__":
    processed_text, extracted_entities = test_complete_pipeline()
    
    print(f"\n🎉 Pipeline test completed successfully!")
    print(f"   Processed text: {len(processed_text.clean_text)} chars")
    print(f"   Extracted entities: {len(extracted_entities)}")