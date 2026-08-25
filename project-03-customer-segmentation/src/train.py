"""Select k, train K-Means, and profile customer segments."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "mall_customers.csv"
MODEL_PATH = PROJECT_DIR / "models" / "customer_segmentation.joblib"
SELECTION_PATH = PROJECT_DIR / "reports" / "k_selection.png"
SEGMENTS_PATH = PROJECT_DIR / "reports" / "customer_segments.png"
PROFILES_PATH = PROJECT_DIR / "reports" / "cluster_profiles.csv"
FEATURES = ["Annual Income (k$)", "Spending Score (1-100)"]
RANDOM_STATE = 42


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Run: python src/download_data.py"
        )

    data = pd.read_csv(DATA_PATH)
    x = data[FEATURES]

    # K-Means uses distances, so both features must use a comparable scale.
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    results: list[dict[str, float | int]] = []
    for k in range(2, 11):
        candidate = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = candidate.fit_predict(x_scaled)
        results.append(
            {
                "k": k,
                "inertia": candidate.inertia_,
                "silhouette": silhouette_score(x_scaled, labels),
            }
        )

    scores = pd.DataFrame(results)
    print("\nCandidate cluster counts:")
    print(scores.round(3).to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.lineplot(data=scores, x="k", y="inertia", marker="o", ax=axes[0])
    axes[0].set_title("Elbow Method")
    axes[0].set_ylabel("Inertia (lower is better)")
    sns.lineplot(data=scores, x="k", y="silhouette", marker="o", ax=axes[1])
    axes[1].set_title("Silhouette Score")
    axes[1].set_ylabel("Score (higher is better)")
    fig.tight_layout()
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SELECTION_PATH, dpi=150)
    plt.close(fig)

    selected_k = 5
    model = KMeans(n_clusters=selected_k, n_init=20, random_state=RANDOM_STATE)
    data["Cluster"] = model.fit_predict(x_scaled)

    profiles = (
        data.groupby("Cluster")
        .agg(
            Customers=("CustomerID", "count"),
            Average_age=("Age", "mean"),
            Average_income=("Annual Income (k$)", "mean"),
            Average_spending_score=("Spending Score (1-100)", "mean"),
        )
        .round(2)
    )
    profiles.to_csv(PROFILES_PATH)
    print(f"\nCluster profiles (k={selected_k}):")
    print(profiles.to_string())

    fig, axis = plt.subplots(figsize=(9, 6))
    sns.scatterplot(
        data=data,
        x=FEATURES[0],
        y=FEATURES[1],
        hue="Cluster",
        palette="tab10",
        s=70,
        ax=axis,
    )
    centers = scaler.inverse_transform(model.cluster_centers_)
    axis.scatter(
        centers[:, 0],
        centers[:, 1],
        marker="X",
        s=250,
        c="black",
        label="Centroids",
    )
    axis.set_title("Mall Customer Segments")
    axis.legend(title="Segment", bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(SEGMENTS_PATH, dpi=150)
    plt.close(fig)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"scaler": scaler, "model": model, "features": FEATURES}, MODEL_PATH
    )
    print(f"\nSaved k-selection chart to: {SELECTION_PATH}")
    print(f"Saved segment chart to: {SEGMENTS_PATH}")
    print(f"Saved profiles to: {PROFILES_PATH}")
    print(f"Saved model artifacts to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
