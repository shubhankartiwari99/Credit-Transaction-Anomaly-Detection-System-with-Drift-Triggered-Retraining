from pathlib import Path

import pandas as pd

DATA_PATH = Path(__file__).resolve().parent / "data" / "creditcard.csv"

df = pd.read_csv(DATA_PATH)

print("Data description:")
print(df.describe())

print("\nClass value counts:")
print(df["Class"].value_counts())
