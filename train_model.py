# train_model.py
# Crop Recommendation with Model Confidence and Out-of-Range Detection
# This file covers steps 3 to 10 of the flow:
#   3 Preprocessing, 4 EDA, 5 Feature selection, 6 Model selection,
#   7 Training, 8 Evaluation, 9 Hyperparameter tuning, 10 Saving the model

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # so plots are saved to files without opening a window
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

np.random.seed(42)

# ---------------------------------------------------------------
# STEP 3: DATA PREPROCESSING
# ---------------------------------------------------------------
print("STEP 3: Data preprocessing")

df = pd.read_csv("Crop_recommendation_5000.csv")
print("Shape of data:", df.shape)

# check missing values and duplicates
print("Missing values:", df.isnull().sum().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()
df = df.dropna()

# The problem statement uses soil nutrients and weather variables,
# so we use only these 7 numeric columns. soil_type is not used.
feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
X = df[feature_names]
y = df["label"]
print("Number of crops (classes):", y.nunique())

# ---------------------------------------------------------------
# STEP 4: EXPLORATORY DATA ANALYSIS (EDA)
# ---------------------------------------------------------------
print("\nSTEP 4: EDA (plots are saved in the outputs folder)")

print(X.describe().round(2))

# count outliers using the IQR rule (we only report them, we do not remove them,
# because different crops really do grow in different extreme conditions)
for col in feature_names:
    q1 = X[col].quantile(0.25)
    q3 = X[col].quantile(0.75)
    iqr = q3 - q1
    count = ((X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)).sum()
    print(col, "-> outliers by IQR rule:", count)

# Plot 1: how many samples each crop has
plt.figure(figsize=(14, 5))
y.value_counts().plot(kind="bar")
plt.title("Number of samples per crop")
plt.tight_layout()
plt.savefig("outputs/eda_class_counts.png")
plt.close()

# Plot 2: histogram of every feature
X.hist(figsize=(12, 8), bins=30)
plt.tight_layout()
plt.savefig("outputs/eda_feature_histograms.png")
plt.close()

# Plot 3: correlation between features
plt.figure(figsize=(7, 6))
sns.heatmap(X.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature correlation")
plt.tight_layout()
plt.savefig("outputs/eda_correlation.png")
plt.close()

# ---------------------------------------------------------------
# STEP 5: FEATURE ENGINEERING
# ---------------------------------------------------------------
print("\nSTEP 5: Feature engineering")
# All 7 features are numeric and have no missing values, so we keep all of them.
# KNN, SVM and Logistic Regression need the features on the same scale,
# so every model below gets a StandardScaler inside its pipeline.
print("Features used:", feature_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training rows:", len(X_train), " Testing rows:", len(X_test))

# ---------------------------------------------------------------
# STEP 6 and 7: MODEL SELECTION AND TRAINING
# ---------------------------------------------------------------
print("\nSTEP 6 and 7: Comparing models")

models = {
    "KNN": Pipeline([("scaler", StandardScaler()), ("model", KNeighborsClassifier())]),
    "Random Forest": Pipeline([("scaler", StandardScaler()), ("model", RandomForestClassifier(n_estimators=200, random_state=42))]),
    "SVM": Pipeline([("scaler", StandardScaler()), ("model", SVC(probability=True, random_state=42))]),
    "Decision Tree": Pipeline([("scaler", StandardScaler()), ("model", DecisionTreeClassifier(random_state=42))]),
    "Logistic Regression": Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression(max_iter=2000))]),
}

results = []
for name, pipe in models.items():
    cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)   # 5-fold cross validation
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    results.append({
        "Model": name,
        "CV Accuracy": round(cv_scores.mean(), 4),
        "Test Accuracy": round(accuracy_score(y_test, pred), 4),
        "Macro F1": round(f1_score(y_test, pred, average="macro"), 4),
    })
    print(name, "done")

results_df = pd.DataFrame(results).sort_values("Macro F1", ascending=False)
print("\n", results_df.to_string(index=False))
results_df.to_csv("outputs/model_comparison.csv", index=False)

best_name = results_df.iloc[0]["Model"]
print("\nBest model by Macro F1:", best_name)

# ---------------------------------------------------------------
# STEP 8: MODEL EVALUATION
# ---------------------------------------------------------------
print("\nSTEP 8: Class-wise evaluation of", best_name)

best_pipe = models[best_name]
pred = best_pipe.predict(X_test)
report = classification_report(y_test, pred, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).T
class_rows = report_df.drop(["accuracy", "macro avg", "weighted avg"])
class_rows.to_csv("outputs/classwise_report.csv")

print("5 weakest crops by F1 score:")
print(class_rows.sort_values("f1-score").head(5)[["precision", "recall", "f1-score", "support"]].round(3))

# Stability test: add small random noise to the test inputs and see
# how many recommendations stay the same.
print("\nStability test (noise = percentage of each feature's std)")
feature_std = X_train.std()
stability_rows = []
for name, pipe in models.items():
    base_pred = pipe.predict(X_test)
    row = {"Model": name}
    for noise_percent in [2, 5, 10]:
        same_list = []
        for repeat in range(5):
            noise = np.random.normal(0, 1, X_test.shape) * feature_std.values * (noise_percent / 100)
            noisy_X = pd.DataFrame(X_test.values + noise, columns=feature_names)
            noisy_pred = pipe.predict(noisy_X)
            same_list.append((noisy_pred == base_pred).mean())
        row["Same at " + str(noise_percent) + "% noise"] = round(np.mean(same_list), 4)
    stability_rows.append(row)
stability_df = pd.DataFrame(stability_rows)
print(stability_df.to_string(index=False))
stability_df.to_csv("outputs/stability_test.csv", index=False)

# ---------------------------------------------------------------
# STEP 9: HYPERPARAMETER TUNING
# ---------------------------------------------------------------
print("\nSTEP 9: Hyperparameter tuning of", best_name)

param_grids = {
    "KNN": {"model__n_neighbors": [3, 5, 7, 9], "model__weights": ["uniform", "distance"]},
    "Random Forest": {"model__n_estimators": [100, 200, 300], "model__max_depth": [None, 15, 25], "model__min_samples_leaf": [1, 2]},
    "SVM": {"model__C": [1, 10, 50], "model__gamma": ["scale", 0.1, 0.5]},
    "Decision Tree": {"model__max_depth": [None, 10, 20], "model__min_samples_leaf": [1, 2, 5]},
    "Logistic Regression": {"model__C": [0.1, 1, 10, 100]},
}

grid = GridSearchCV(models[best_name], param_grids[best_name], cv=5, scoring="f1_macro")
grid.fit(X_train, y_train)
print("Best parameters:", grid.best_params_)
print("Best CV macro F1:", round(grid.best_score_, 4))

final_model = grid.best_estimator_
final_pred = final_model.predict(X_test)
print("Test accuracy after tuning:", round(accuracy_score(y_test, final_pred), 4))
print("Test macro F1 after tuning:", round(f1_score(y_test, final_pred, average="macro"), 4))

# ---------------------------------------------------------------
# CONFIDENCE: how sure is the model?
# ---------------------------------------------------------------
print("\nModel confidence check")
CONFIDENCE_LIMIT = 0.50      # below this we say "low confidence"
probs = final_model.predict_proba(X_test)
top_conf = probs.max(axis=1)
correct = (final_pred == y_test.values)
print("Average confidence when correct:", round(top_conf[correct].mean(), 3))
print("Average confidence when wrong:  ", round(top_conf[~correct].mean(), 3))
sure = top_conf >= CONFIDENCE_LIMIT
print("Predictions with confidence >=", CONFIDENCE_LIMIT, ":", round(sure.mean() * 100, 1), "%")
print("Accuracy on those predictions:", round(correct[sure].mean(), 4))

# ---------------------------------------------------------------
# OUT-OF-RANGE DETECTION (unusual combinations)
# ---------------------------------------------------------------
# Idea: for a new input, find its 5 nearest training rows (after scaling).
# If it is much farther from them than normal training rows are from
# each other, the combination is unusual and we do NOT recommend a crop.
print("\nOut-of-range detector")

ood_scaler = StandardScaler()
X_train_scaled = ood_scaler.fit_transform(X_train)

K_NEIGHBORS = 5
finder = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1)   # +1 because the first neighbor is the row itself
finder.fit(X_train_scaled)
dist, _ = finder.kneighbors(X_train_scaled)
train_scores = dist[:, 1:].mean(axis=1)                  # skip the row itself

PERCENTILE = 99
distance_limit = np.percentile(train_scores, PERCENTILE)
print("Distance limit (", PERCENTILE, "th percentile of training rows):", round(distance_limit, 3))

finder_for_new = NearestNeighbors(n_neighbors=K_NEIGHBORS)
finder_for_new.fit(X_train_scaled)

def unusual_score(data):
    # average distance to the 5 nearest training rows
    scaled = ood_scaler.transform(data)
    d, _ = finder_for_new.kneighbors(scaled)
    return d.mean(axis=1)

# Check 1: real test rows should rarely be flagged
test_scores = unusual_score(X_test)
print("Real test rows flagged as unusual:", round((test_scores > distance_limit).mean() * 100, 1), "%")

# Check 2: fake rows made by mixing values from different rows.
# Each feature is taken from a different random row, so combinations such as
# very high rainfall with very low humidity can appear.
fake = pd.DataFrame({col: np.random.choice(X_train[col].values, 1000) for col in feature_names})
fake_scores = unusual_score(fake)
print("Mixed-up fake combinations flagged:", round((fake_scores > distance_limit).mean() * 100, 1), "%")

# Check 3: random values spread uniformly between min and max of each feature
low = X_train.min().values
high = X_train.max().values
random_rows = pd.DataFrame(np.random.uniform(low, high, (1000, len(feature_names))), columns=feature_names)
random_scores = unusual_score(random_rows)
print("Uniform random rows flagged:", round((random_scores > distance_limit).mean() * 100, 1), "%")

# Save the numbers to a small file for the report
ood_summary = pd.DataFrame({
    "Test": ["Real test rows (should be low)", "Mixed-up fake combinations", "Uniform random rows"],
    "Flagged %": [
        round((test_scores > distance_limit).mean() * 100, 1),
        round((fake_scores > distance_limit).mean() * 100, 1),
        round((random_scores > distance_limit).mean() * 100, 1),
    ],
})
ood_summary.to_csv("outputs/out_of_range_test.csv", index=False)

# ---------------------------------------------------------------
# STEP 10: SAVING THE MODEL
# ---------------------------------------------------------------
print("\nSTEP 10: Saving the model")

# Train the final model once more on ALL the data, then save it with
# everything the app needs.
final_model.fit(X, y)
ood_scaler_full = StandardScaler()
X_scaled_full = ood_scaler_full.fit_transform(X)
finder_full = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1).fit(X_scaled_full)
dist_full, _ = finder_full.kneighbors(X_scaled_full)
limit_full = np.percentile(dist_full[:, 1:].mean(axis=1), PERCENTILE)
finder_full_new = NearestNeighbors(n_neighbors=K_NEIGHBORS).fit(X_scaled_full)

bundle = {
    "model_name": best_name,
    "model": final_model,
    "feature_names": feature_names,
    "ood_scaler": ood_scaler_full,
    "ood_finder": finder_full_new,
    "distance_limit": float(limit_full),
    "confidence_limit": CONFIDENCE_LIMIT,
    "train_min": X.min().to_dict(),
    "train_max": X.max().to_dict(),
}
joblib.dump(bundle, "crop_model.pkl")
print("Saved crop_model.pkl")
print("\nAll done.")
