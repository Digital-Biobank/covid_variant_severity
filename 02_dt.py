# %% Imports
import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, \
    f1_score

# %% Read in combined clusters, PCA transformed, and mortality data
df = pd.read_parquet(
    "02_77142-vcf_"
    "2-component-pca-transformed_"
    "3-cluster-kmeans_"
    "outcomes.parquet"
)

# %% Convert cluster data into indicator (dummy) variables
dummy_df = pd.get_dummies(df, columns=["cluster"])

# %% Prepare data for scikit-learn
Xy = dummy_df[[
    "covv_patient_age",
    "gender",
    "cluster_0",
    "cluster_1",
    "is_red"
]].dropna()

Xy.to_parquet("03_77142-vcf_2-component-pca_3-cluster-kmeans_outcomes_dropna.pickle")

X = Xy.drop("is_red", axis=1)
y = Xy["is_red"]

# %% Fit cluster model with scikit-learn
clf = DecisionTreeClassifier()
clf.fit(X, y)
joblib.dump(clf, "03_77142-vcf_2-component-pca_3-cluster-kmeans_rf.pickle")


# %% Use sklearn logisitic regression model for prediction
pred = clf.predict(X)

# %% Show model metrics
print(classification_report(y, pred))
print(confusion_matrix(y, pred))
precision_score(y, pred)
recall_score(y, pred)
f1_score(y, pred)

# %% Manually calculate model metrics (work in progress)
true_pred = pd.DataFrame({"actual": y, "predicted": pred}).sort_values(["predicted"])
true_pred = true_pred.assign(
    true_positives=true_pred["actual"].cumsum(),
    total_positives=true_pred["actual"].sum(),
)

true_pred = true_pred.assign(
    recall=true_pred["true_positives"] / true_pred["total_positives"],
    predicted_positives=range(1, len(true_pred) + 1),
)
true_pred = true_pred.assign(
    precision=true_pred["true_positives"] / true_pred["predicted_positives"],
    false_positives=true_pred["predicted_positives"] - true_pred["true_positives"],
    total_negatives=len(true_pred) - true_pred["total_positives"],
)
true_pred = true_pred.assign(
    fallout=true_pred["false_positives"] / true_pred["total_negatives"]
)
