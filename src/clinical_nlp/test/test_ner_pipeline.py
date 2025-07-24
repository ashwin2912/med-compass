#!/usr/bin/env python3
"""
Test the complete preprocessing + NER + Topic Modeling pipeline
"""

import sys
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from ..nlp.preprocessing import preprocess_clinical_text
from ..nlp.entity_extraction import extract_clinical_entities
from ..nlp.topic_modeling import ClinicalTopicModeler, analyze_clinical_topics

def test_complete_pipeline():
    """Test preprocessing + NER + Topic Modeling on sample clinical texts"""
    
    # Multiple sample MIMIC-style clinical notes for topic modeling
    sample_notes = [
        """
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
        """,
        
        """
        ADMISSION DIAGNOSIS: Community-acquired pneumonia

        HISTORY OF PRESENT ILLNESS:
        65-year-old female with COPD presented with 3-day history of increased dyspnea, 
        productive cough with yellow sputum, and fever.

        PHYSICAL EXAMINATION:
        Vital signs: Temperature 101.2F, BP 130/85, HR 95, RR 22, O2 sat 88% on room air
        Pulmonary: Crackles bilateral lower lobes, decreased breath sounds

        DIAGNOSTIC STUDIES:
        Chest X-ray: Right lower lobe infiltrate
        White blood cell count: 15,000

        TREATMENT:
        Started on ceftriaxone and azithromycin
        Oxygen therapy initiated
        """,
        
        """
        ADMISSION DIAGNOSIS: Diabetic ketoacidosis

        HISTORY OF PRESENT ILLNESS:
        28-year-old male with type 1 diabetes mellitus presented with nausea, vomiting, 
        and altered mental status. Patient ran out of insulin 2 days ago.

        PHYSICAL EXAMINATION:
        Vital signs: BP 110/70, HR 110, RR 26, Temperature 98.9F
        General: Appears dehydrated, fruity breath odor

        LABORATORY RESULTS:
        Glucose: 485 mg/dL
        Ketones: Large
        pH: 7.25
        Bicarbonate: 12 mEq/L

        TREATMENT:
        IV insulin drip initiated
        Normal saline resuscitation
        Electrolyte monitoring
        """,
        
        """
        ADMISSION DIAGNOSIS: Acute kidney injury

        HISTORY OF PRESENT ILLNESS:
        72-year-old male with chronic kidney disease stage 3 presented with decreased urine output
        and lower extremity edema. Recently started on ibuprofen for arthritis pain.

        PHYSICAL EXAMINATION:
        Vital signs: BP 160/95, HR 88, RR 18, afebrile
        Extremities: 2+ pitting edema bilateral lower extremities

        LABORATORY RESULTS:
        Creatinine: 3.2 mg/dL (baseline 1.8)
        BUN: 45 mg/dL
        Potassium: 5.1 mEq/L

        TREATMENT:
        Discontinued NSAIDs
        Fluid restriction
        Nephrology consultation
        """,
        
        """
        ADMISSION DIAGNOSIS: Sepsis secondary to UTI

        HISTORY OF PRESENT ILLNESS:
        82-year-old female nursing home resident presented with altered mental status,
        fever, and decreased oral intake. Foley catheter in place.

        PHYSICAL EXAMINATION:
        Vital signs: Temperature 102.1F, BP 90/50, HR 120, RR 24
        General: Lethargic, oriented to person only

        DIAGNOSTIC STUDIES:
        Urinalysis: >50 WBC, nitrites positive, bacteria present
        Blood cultures: Pending
        Lactate: 2.8

        TREATMENT:
        IV fluids for hypotension
        Empiric antibiotics started
        Foley catheter removed
        """
    ]
    
    print("🔄 Testing Complete NLP Pipeline with Topic Modeling")
    print("=" * 70)
    
    # Process each note individually for detailed analysis
    processed_notes = []
    all_entities = []
    
    print("Step 1: Processing Individual Clinical Notes...")
    print("-" * 50)
    
    for i, note in enumerate(sample_notes):
        print(f"\n📄 Processing Note {i+1}...")
        
        # Preprocessing
        processed = preprocess_clinical_text(note)
        processed_notes.append(processed.clean_text)
        
        # Entity extraction
        entities = extract_clinical_entities(processed.clean_text)
        all_entities.append(entities)
        
        print(f"   ✅ Clean text: {len(processed.clean_text)} chars")
        print(f"   ✅ Entities found: {len(entities)}")
        print(f"   ✅ Sections: {list(processed.sections.keys())}")
    
    # Step 2: Topic Modeling on all processed notes
    print(f"\nStep 2: Topic Modeling Analysis...")
    print("-" * 50)
    
    try:
        # Perform topic modeling
        topic_modeler, doc_topics = analyze_clinical_topics(
            documents=processed_notes, 
            n_topics=3  # Using 3 topics for our 5 documents
        )
        
        # Display topics
        topic_modeler.print_topics(top_n=8)
        
        # Show document-topic associations
        print(f"\n📊 Document-Topic Associations:")
        print("-" * 40)
        
        for i, doc_topic in enumerate(doc_topics):
            dominant_topic = doc_topic.dominant_topic
            dominant_weight = doc_topic.dominant_topic_weight
            
            print(f"\n📄 Note {i+1}:")
            print(f"   Dominant Topic: #{dominant_topic + 1} (weight: {dominant_weight:.3f})")
            print(f"   All topics: ", end="")
            for topic_id, weight in doc_topic.topic_distributions.items():
                print(f"T{topic_id+1}:{weight:.2f} ", end="")
            print()
        
        # Get topic summary as DataFrame
        topic_summary = topic_modeler.get_topic_summary()
        print(f"\n📋 Topic Summary:")
        print("-" * 40)
        print(topic_summary.to_string(index=False))
        
    except Exception as e:
        print(f"❌ Topic modeling failed: {e}")
        topic_modeler = None
        doc_topics = None
    
    # Step 3: Combined Analysis Results
    print(f"\n📈 Combined Pipeline Results:")
    print("-" * 40)
    
    # Entity analysis across all documents
    entity_counts = {}
    total_entities = 0
    
    for entities in all_entities:
        total_entities += len(entities)
        for entity in entities:
            entity_type = entity.label.value
            entity_counts[entity_type] = entity_counts.get(entity_type, 0) + 1
    
    print(f"📊 Entity Statistics:")
    print(f"   Total entities: {total_entities}")
    print(f"   Entity types: {len(entity_counts)}")
    
    for entity_type, count in sorted(entity_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"   {entity_type}: {count}")
    
    # Show sample entities from first note
    if all_entities:
        print(f"\n🔍 Sample Entities from First Note:")
        sample_entities = all_entities[0][:5]  # First 5 entities
        for entity in sample_entities:
            print(f"   • '{entity.text}' → {entity.label.value} (conf: {entity.confidence:.2f})")
    
    return processed_notes, all_entities, topic_modeler, doc_topics

def demonstrate_topic_insights(topic_modeler, doc_topics, processed_notes):
    """Demonstrate clinical insights from topic modeling"""
    
    if not topic_modeler or not doc_topics:
        print("❌ Cannot demonstrate insights - topic modeling failed")
        return
    
    print(f"\n🧠 Clinical Insights from Topic Analysis:")
    print("=" * 60)
    
    # Group documents by dominant topic
    topic_groups = {}
    for i, doc_topic in enumerate(doc_topics):
        topic_id = doc_topic.dominant_topic
        if topic_id not in topic_groups:
            topic_groups[topic_id] = []
        topic_groups[topic_id].append(i)
    
    print(f"📋 Patient Clustering by Clinical Topics:")
    for topic_id, doc_indices in topic_groups.items():
        topic = topic_modeler.topics[topic_id]
        print(f"\n🏥 {topic.description}")
        print(f"   Documents: {len(doc_indices)} ({', '.join([f'Note {i+1}' for i in doc_indices])})")
        print(f"   Key themes: {', '.join(topic.keywords[:4])}")
    
    # Find documents with mixed topic distributions
    print(f"\n🔄 Complex Cases (Mixed Topics):")
    for i, doc_topic in enumerate(doc_topics):
        # Find cases where dominant topic weight is < 0.6 (indicating mixed themes)
        if doc_topic.dominant_topic_weight < 0.6:
            print(f"   📄 Note {i+1}: Mixed presentation")
            sorted_topics = sorted(doc_topic.topic_distributions.items(), 
                                 key=lambda x: x[1], reverse=True)[:2]
            for topic_id, weight in sorted_topics:
                topic_desc = topic_modeler.topics[topic_id].description
                print(f"      {topic_desc}: {weight:.2f}")

if __name__ == "__main__":
    print("🚀 Starting Complete Clinical NLP Pipeline Test")
    
    processed_texts, entities_list, modeler, doc_topics = test_complete_pipeline()
    
    # Demonstrate clinical insights
    demonstrate_topic_insights(modeler, doc_topics, processed_texts)
    
    print(f"\n🎉 Pipeline test completed successfully!")
    print(f"   📄 Processed documents: {len(processed_texts)}")
    print(f"   🏷️  Total entities extracted: {sum(len(entities) for entities in entities_list)}")
    if modeler:
        print(f"   📋 Topics discovered: {len(modeler.topics)}")
        print(f"   🔍 Topic modeling: ✅ Success")
    else:
        print(f"   🔍 Topic modeling: ❌ Failed")