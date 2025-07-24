from .ingestion import ingest_mimic_data


def ingest_data():
# Load MIMIC-III data with defaults
    notes, metadata = ingest_mimic_data(
        file_path="/Users/ashwindhanasamy/Documents/cave/experiments/health-tech/clinical-nlp-pipeline/data/mimic-iii-clinical-database-demo-1.4/NOTEEVENTS.csv",
        sample_size=1000,  # For testing
        categories=["Discharge summary", "Physician", "Nursing"]
    )

    print(f"Loaded {len(notes)} valid clinical notes")
    print(f"Success rate: {metadata.success_rate:.1%}")
    print(notes)

def main():
    ingest_data()


if __name__ == '__main__':
    main()