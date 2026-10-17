
from pathlib import Path

import pandas as pd

# Dataset location
DATA_PATH = Path("data/raw/creditcard.csv")


def main():
    if not DATA_PATH.exists():
        print(f"ERROR: Dataset not found at {DATA_PATH.resolve()}")
        print("Please place creditcard.csv inside data/raw/")
        return

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("\n" + "=" * 60)
    print("DATASET OVERVIEW")
    print("=" * 60)

    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset information:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print(f"\nDuplicate rows: {df.duplicated().sum():,}")

    if "Class" in df.columns:
        print("\nTransaction class distribution:")
        print(df["Class"].value_counts())

        print("\nClass percentages:")
        print(
            (df["Class"].value_counts(normalize=True) * 100)
            .round(4)
            .astype(str)
            .add("%")
        )

        print("\nFraud rate:")
        print(f"{df['Class'].mean() * 100:.4f}%")

    print("\nDataset inspection completed.")


if __name__ == "__main__":
    main()
