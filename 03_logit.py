# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, plot_roc_curve, confusion_matrix, recall_score, precision_score, \
    f1_score

# %% Read in combined clusters, PCA transformed, and mortality data
mort = pd.read_parquet(
    "02_77142-vcf_"
    "2-component-pca-transformed_"
    "mortality_"
    "3-cluster-kmeans"
    ".parquet"
)

# %% Convert cluster data into indicator (dummy) variables
dummy_df = pd.get_dummies(mort, columns=["cluster"])

# %% Fit base model with statsmodels
base_mod = smf.logit(formula="y ~ covv_patient_age + gender", data=dummy_df)
base_fit = base_mod.fit()
base_fit.summary()

# %% Fit cluster model with statsmodels
clus_mod = smf.logit(formula="y ~ covv_patient_age + gender + cluster_0 + cluster_1", data=dummy_df)
clus_fit = clus_mod.fit()
clus_fit.summary()

# %% Fit principal component model with statsmodels
comp_mod = smf.logit(formula="y ~ covv_patient_age + gender + PC1 + PC2", data=dummy_df)
comp_fit = comp_mod.fit()
comp_fit.summary()

# %% Prepare data for scikit-learn
Xy = dummy_df[[
    "covv_patient_age",
    "gender",
    "cluster_0",
    "cluster_1",
    "y"
]].dropna()

X = Xy.drop("y", axis=1)
y = Xy["y"]

# %% Fit base model with scikit-learn
base_lr = LogisticRegression(penalty='none')
base_lr.fit(X[["covv_patient_age", "gender"]], y)

# %% Fit cluster model with scikit-learn
clus_lr = LogisticRegression(penalty='none')
clus_lr.fit(X, y)

# %% Plot ROC curves
plot_roc_curve(base_lr, X[["covv_patient_age", "gender"]], y)
plot_roc_curve(clus_lr, X, y)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.savefig(
    "02_77142-vcf_2-component-pca-transformed_"
    "mortality_3-cluster-kmeans_"
    "logisitic-regression_roc-curve.png"
)
plt.show()

# %% Use sklearn logisitic regression model for prediction
pred = clus_lr.predict(X)

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
