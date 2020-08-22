import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score

# %% plot roc curves
xy = pd.read_parquet("03_77142-vcf_2-component-pca_3-cluster-kmeans_outcomes_dropna.pickle")
clus_lr_master = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_logistic-regression-model.pickle")
clus_lr_random = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_logistic-regression-model_random.pickle")

x = xy.drop("is_red", axis=1)
y = xy["is_red"]
clfs = [clus_lr_master, clus_lr_random]
labs = ["Logisitic Regression", "Logisitic Regression (random)"]
for lab, clf in zip(labs, clfs):
    pred = clf.predict_proba(x)[::, 1]
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
