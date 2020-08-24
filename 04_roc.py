import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import auc, roc_curve, roc_auc_score

# %% Plot ROC curves
df = pd.read_parquet("03_77142-vcf_2-component-pca_3-cluster-kmeans_outcomes_dropna.pickle")
df_random = pd.read_parquet("03_77142-vcf_2-component-pca_3-cluster-kmeans_outcomes_dropna_random.pickle")
lr_master = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_logistic-regression-model.pickle")
lr_random = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_logistic-regression-model_random.pickle")

cv = StratifiedKFold(n_splits=5)

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
    X = df.drop("is_red", axis=1)
    y = df["is_red"]
    pred = clf.predict_proba(X)[::, 1]
    fpr, tpr, _ = roc_curve(y, pred)
    auc = roc_auc_score(y, pred)
    plt.plot(fpr, tpr, label=f"{lab}, AUC={auc:.3f}")
    plt.legend(loc=4)
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.savefig(
    "02_77142-vcf_2-component-pca-transformed_"
    "mortality_3-cluster-kmeans_"
    "knn_roc-curve.png"
)
plt.show()

dt_master = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_dt.pickle")
dt_random = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_dt_random.pickle")

clfs = [dt_master, dt_random]
labs = ["Decision Tree", "Decision Tree (random)"]
for clf, df, lab in zip(clfs, dfs, labs):
    X = df.drop("is_red", axis=1)
    y = df["is_red"]
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
    "decision-tree_roc-curve.png"
)
plt.show()

rf_master = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_rf.pickle")
rf_random = joblib.load("03_77142-vcf_2-component-pca_3-cluster-kmeans_rf_random.pickle")

clfs = [rf_master, rf_random]
labs = ["Random Forest", "Random Forest (random)"]
for clf, df, lab in zip(clfs, dfs, labs):
    X = df.drop("is_red", axis=1)
    y = df["is_red"]
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
    "random-forest_roc-curve.png"
)
plt.show()


# classifier = dt_master
#
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
#
# fig, ax = plt.subplots()
# for i, (train, test) in enumerate(cv.split(X, y)):
#     classifier.fit(X[train], y[train])
#     viz = plot_roc_curve(classifier, X[test], y[test],
#                          name='ROC fold {}'.format(i),
#                          alpha=0.3, lw=1, ax=ax)
#     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(viz.roc_auc)
#
# ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#         label='Chance', alpha=.8)
#
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(mean_fpr, mean_tpr, color='b',
#         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#         lw=2, alpha=.8)
#
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                 label=r'$\pm$ 1 std. dev.')
#
# ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05],
#        title="Receiver operating characteristic example")
# ax.legend(loc="lower right")
# plt.show()