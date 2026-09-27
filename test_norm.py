import pandas as pd
from src.preprocessing.normalize import Normalizer

# Create a small dummy dataframe that hits all our noise patterns
data = {
    "entity_id": ["S1-1", "S2-1", "S3-1", "S2-2", "S2-3"],
    "business_name": [
        "ABC Technologies",
        "ABC TECHNOLOGIES PVT. LTD.",
        "ABC Technologies & Co.",
        "मॉडर्न फाइनेंस", # Devanagari
        "École de Paris SARL", # French accents
    ],
    "business_address": [
        "123 Main St, New York, NY",
        "123 Main Street, New York, NY",
        "123 Main St., New York, NY",
        "No 10 Enkay Square, Udyog Vihar Phase V",
        "63 R. DE DIEPPE, LILLE"
    ],
    "country": ["US", "US", "US", "India", "France"]
}

df = pd.DataFrame(data)

print("--- Original Data ---")
print(df[["business_name", "business_address", "country"]])

norm = Normalizer()
norm_df = norm.normalize_dataframe(df)

print("\n--- Normalized Data ---")
for _, row in norm_df.iterrows():
    print(f"ID: {row['entity_id']} ({row['country']})")
    print(f"  Orig Name: {row['original_name']}")
    print(f"  Norm Name: {row['normalized_name']}")
    print(f"  Orig Addr: {row['original_address']}")
    print(f"  Norm Addr: {row['normalized_address']}")
    print(f"  Name without suffix: {row['name_without_legal_suffix']}")
    print(f"  Addr Numbers: {row['numbers']}")
    print()
