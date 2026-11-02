import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from data_loader import load_split, load_labels


# -----------------------------------------
# 1. Load the dataset
# -----------------------------------------

print("Loading dataset...")

labels = load_labels()

X_train, y_train, subjects_train = load_split("train")
X_test, y_test, subjects_test = load_split("test")

activity_names = y_train.map(labels)

print("\n--- DATASET INFORMATION ---")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

print("Missing training values:", X_train.isnull().sum().sum())
print("Missing testing values:", X_test.isnull().sum().sum())

print("Training volunteers:", subjects_train.nunique())
print("Testing volunteers:", subjects_test.nunique())


# -----------------------------------------
# 2. Activity distribution
# -----------------------------------------

print("\nGenerating activity distribution graph...")

counts = activity_names.value_counts()

plt.figure(figsize=(10, 5))

sns.barplot(
    x=counts.index,
    y=counts.values
)

plt.title("Distribution of Human Activities")
plt.xlabel("Activity")
plt.ylabel("Number of Samples")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    "reports/figures/class_distribution.png",
    dpi=150
)

plt.close()


# -----------------------------------------
# 3. Sensor feature boxplot
# -----------------------------------------

print("Generating sensor feature boxplot...")

plt.figure(figsize=(10, 5))

sns.boxplot(
    x=activity_names,
    y=X_train["tBodyAccMag-std()"]
)

plt.title("Body Acceleration Variation by Activity")
plt.xlabel("Activity")
plt.ylabel("Body Acceleration Magnitude Standard Deviation")
plt.xticks(rotation=30, ha="right")

plt.tight_layout()

plt.savefig(
    "reports/figures/feature_boxplot.png",
    dpi=150
)

plt.close()


# -----------------------------------------
# 4. PCA visualization
# -----------------------------------------

print("Generating PCA scatter plot...")

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X_train)

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=activity_names,
    s=10,
    alpha=0.7
)

plt.title("PCA Visualization of Human Activities")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.tight_layout()

plt.savefig(
    "reports/figures/pca_scatter.png",
    dpi=150
)

plt.close()


# -----------------------------------------
# 5. Display summary
# -----------------------------------------

print("\n--- ACTIVITY DISTRIBUTION ---")
print(counts)

print("\n--- PCA INFORMATION ---")

explained_variance = pca.explained_variance_ratio_.sum()

print(
    "Variance explained by two components:",
    round(explained_variance * 100, 2),
    "%"
)

print("\nEDA completed successfully!")
print("All graphs saved in reports/figures/")