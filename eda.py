
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "raw" / "creditcard.csv"
REPORT_DIR = BASE_DIR / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")


def save_plot(filename):
    """Save the current figure and close it."""
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Saved: reports/figures/{filename}")


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Place creditcard.csv inside data/raw/."
        )

    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    # --------------------------------------------------------
    # 1. DATA QUALITY SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATA QUALITY ANALYSIS")
    print("=" * 60)

    quality = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "missing_values": df.isnull().sum(),
        "unique_values": df.nunique(),
    })

    quality.to_csv(REPORT_DIR / "data_quality.csv")
    print(quality)

    duplicate_count = int(df.duplicated().sum())
    duplicate_fraud_counts = (
        df.loc[df.duplicated(keep=False), "Class"]
        .value_counts()
        .sort_index()
    )

    print(f"\nExact duplicate rows (excluding first occurrences): "
          f"{duplicate_count:,}")
    print("\nClass counts among rows belonging to duplicate groups:")
    print(duplicate_fraud_counts)

    # --------------------------------------------------------
    # 2. CLASS DISTRIBUTION
    # --------------------------------------------------------

    class_counts = df["Class"].value_counts().sort_index()
    fraud_rate = df["Class"].mean() * 100

    print("\nTransaction counts:")
    print(f"Legitimate: {class_counts.get(0, 0):,}")
    print(f"Fraudulent: {class_counts.get(1, 0):,}")
    print(f"Fraud rate: {fraud_rate:.4f}%")

    class_summary = pd.DataFrame({
        "class": ["Legitimate", "Fraudulent"],
        "count": [
            int(class_counts.get(0, 0)),
            int(class_counts.get(1, 0)),
        ],
    })
    class_summary["percentage"] = (
        class_summary["count"] / len(df) * 100
    )
    class_summary.to_csv(
        REPORT_DIR / "class_distribution.csv", index=False
    )

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        data=class_summary,
        x="class",
        y="count",
    )
    ax.set_title("Legitimate vs Fraudulent Transactions")
    ax.set_xlabel("Transaction Class")
    ax.set_ylabel("Number of Transactions")

    for container in ax.containers:
        ax.bar_label(container, fmt="%.0f")

    save_plot("class_distribution.png")

    # --------------------------------------------------------
    # 3. TRANSACTION AMOUNT DISTRIBUTION
    # --------------------------------------------------------

    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="Amount",
        bins=80,
        stat="count",
    )
    plt.title("Transaction Amount Distribution")
    plt.xlabel("Transaction Amount")
    plt.ylabel("Number of Transactions")
    save_plot("amount_distribution.png")

    # Log scale makes the long-tailed amount distribution
    # easier to inspect without removing large transactions.
    plt.figure(figsize=(10, 5))
    sns.histplot(
        data=df,
        x="Amount",
        bins=80,
        log_scale=(True, False),
    )
    plt.title("Transaction Amount Distribution (Log X Scale)")
    plt.xlabel("Transaction Amount — Log Scale")
    plt.ylabel("Number of Transactions")
    save_plot("amount_distribution_log.png")

    # --------------------------------------------------------
    # 4. AMOUNT BY CLASS
    # --------------------------------------------------------

    amount_summary = df.groupby("Class")["Amount"].agg(
        ["count", "mean", "median", "std", "min", "max"]
    )
    amount_summary.index = amount_summary.index.map({
        0: "Legitimate",
        1: "Fraudulent",
    })
    amount_summary.to_csv(REPORT_DIR / "amount_by_class.csv")

    print("\nAmount statistics by class:")
    print(amount_summary)

    plt.figure(figsize=(9, 5))
    sns.boxplot(
        data=df,
        x="Class",
        y="Amount",
        showfliers=False,
    )
    plt.title("Transaction Amount by Class")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Amount")
    save_plot("amount_by_class.png")

    # --------------------------------------------------------
    # 5. FRAUD OVER TIME
    # --------------------------------------------------------

    # Time is elapsed seconds from the dataset's first recorded
    # transaction, not a real-world timestamp.
    time_df = df[["Time", "Class"]].copy()
    time_df["TimeWindow"] = (
        time_df["Time"] // 3600
    ).astype(int)

    hourly = time_df.groupby("TimeWindow").agg(
        transactions=("Class", "size"),
        frauds=("Class", "sum"),
    )
    hourly["fraud_rate_pct"] = (
        hourly["frauds"] / hourly["transactions"] * 100
    )
    hourly.to_csv(REPORT_DIR / "fraud_by_time_window.csv")

    plt.figure(figsize=(12, 5))
    plt.plot(
        hourly.index,
        hourly["fraud_rate_pct"],
        marker="o",
        markersize=3,
    )
    plt.title("Fraud Rate by Elapsed-Time Window")
    plt.xlabel("Elapsed Time Window (hours)")
    plt.ylabel("Fraud Rate (%)")
    save_plot("fraud_rate_over_time.png")

    # --------------------------------------------------------
    # 6. FEATURE CORRELATION WITH CLASS
    # --------------------------------------------------------

    correlations = (
        df.corr(numeric_only=True)["Class"]
        .drop("Class")
        .sort_values(key=lambda s: s.abs(), ascending=False)
    )

    correlations.rename("correlation_with_class").to_csv(
        REPORT_DIR / "feature_correlations.csv"
    )

    print("\nTop 10 features by absolute correlation with Class:")
    print(correlations.head(10))

    top_features = correlations.head(10).index.tolist()
    correlation_data = df[top_features + ["Class"]].corr()

    plt.figure(figsize=(11, 8))
    sns.heatmap(
        correlation_data,
        cmap="coolwarm",
        center=0,
        annot=True,
        fmt=".2f",
    )
    plt.title("Correlation of Top Features with Fraud Class")
    save_plot("feature_correlation.png")

    # --------------------------------------------------------
    # 7. FINAL SUMMARY
    # --------------------------------------------------------

    summary = {
        "total_transactions": len(df),
        "legitimate_transactions": int((df["Class"] == 0).sum()),
        "fraudulent_transactions": int((df["Class"] == 1).sum()),
        "fraud_rate_percent": round(float(fraud_rate), 6),
        "missing_cells": int(df.isnull().sum().sum()),
        "exact_duplicate_rows": duplicate_count,
        "total_transaction_amount": round(float(df["Amount"].sum()), 2),
    }

    pd.Series(summary, name="value").to_csv(
        REPORT_DIR / "eda_summary.csv"
    )

    print("\n" + "=" * 60)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 60)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nReports saved to: {REPORT_DIR}")


if __name__ == "__main__":
    main()
