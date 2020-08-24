import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# %% Plot ROC curves
df = pd.read_parquet("03_77142-vcf_2-component-pca_3-cluster-kmeans_outcomes_dropna.pickle")
df_random = pd.read_parquet("03_77142-vcf_2-component-pca_3-cluster-kmeans_outcomes_dropna_random.pickle")
lr_master = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_logistic-regression-model.pickle")
lr_random = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_logistic-regression-model_random.pickle")

clfs = lr_master, lr_random
dfs = df, df_random
labs = ["Logistic Regression", "Logistic Regression (random)"]
for clf, df, lab in zip(clfs, dfs, labs):
    X = df.drop("is_red", axis=1)
    y = df["is_red"]
    pred = clf.predict_proba(X)[::, 1]
    fpr, tpr, _ = roc_curve(y, pred)
    auc = roc_auc_score(y, pred)
    plt.plot(fpr, tpr, label=f"{lab}, AUC={auc:.3f}")
    plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.savefig(
    "02_77142-vcf_2-component-pca-transformed_"
    "mortality_3-cluster-kmeans_"
    "logisitic-regression_roc-curve.png"
)
plt.show()

knn_master = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_knn.pickle")
knn_random = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_knn_random.pickle")

clfs = [knn_master, knn_random]
labs = ["K nearest neighbors", "K nearest neighbors (random)"]
for clf, df, lab in zip(clfs, dfs, labs):
    pred = clf.predict_proba(X)[::, 1]
    fpr, tpr, _ = roc_curve(y, pred)
    auc = roc_auc_score(y, pred)
    plt.plot(fpr, tpr, label=f"{lab}, AUC={auc:.3f}")
    plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='chance', alpha=.8)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig(
    "02_77142-vcf_2-component-pca-transformed_"
    "mortality_3-cluster-kmeans_"
    "logisitic-regression_roc-curve.png"
)
plt.show()

dt_master = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_dt.pickle")
dt_random = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_dt_random.pickle")

clfs = [dt_master, dt_random]
labs = ["K nearest neighbors", "K nearest neighbors (random)"]
for clf, df, lab in zip(clfs, dfs, labs):
    pred = clf.predict_proba(X)[::, 1]
    fpr, tpr, _ = roc_curve(y, pred)
    auc = roc_auc_score(y, pred)
    plt.plot(fpr, tpr, label=f"{lab}, AUC={auc:.3f}")
    plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='chance', alpha=.8)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig(
    "02_77142-vcf_2-component-pca-transformed_"
    "mortality_3-cluster-kmeans_"
    "logisitic-regression_roc-curve.png"
)
plt.show()
