# %% Imports
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import statsmodels as sm
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, \
    f1_score
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

# %% Read in combined clusters, PCA transformed, and mortality data
df = pd.read_parquet(
    "data/02_77142-vcf_9-component-pca-transformed_"
    "3-cluster-kmeans.parquet",
)
df = df[[
    "is_red",
    "covv_patient_age",
    "gender",
    "clade",
    "cat_region",
    "cluster"
]].dropna().astype({
    "is_red": int,
    "gender": int,
    "clade": int,
    "cat_region": int
})

# %% Convert categorical data into indicator (dummy) variables
df = pd.get_dummies(df, columns=["cat_region", "clade", "cluster"], drop_first=True)

X = df.dropna()
y = X["is_red"]
X = X.drop(["is_red"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)
logreg = LogisticRegression(penalty='none', max_iter=1e4)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
roc_auc_score(y_test, pred)

# %% Prepare data for scikit-learn
df = pd.read_parquet("data/02_semi_knn-kernel-label-spreading-dataframe.parquet")
df = df[[
    "is_red",
    "covv_patient_age",
    "gender",
    "clade",
    "cat_region",
    "spread_labels"
]].dropna().astype({
    "is_red": int,
    "gender": int,
    "clade": int,
    "cat_region": int,
    "spread_labels": int,
})
true_X = df.dropna()
true_y = true_X["is_red"]
true_X = true_X.drop(["is_red", "spread_labels"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    true_X,
    true_y,
    test_size=0.33,
    random_state=42
)


logreg = LogisticRegression(penalty='none', max_iter=1e4)
logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
roc_auc_score(y_test, pred)
accuracy_score(y_test, pred)

pred_X = df.drop(y_test.index)
pred_y = pred_X["spread_labels"]
pred_X = pred_X.drop(["is_red", "spread_labels"], axis=1)
logreg = LogisticRegression(penalty='none', max_iter=1e4)
logreg.fit(pred_X, pred_y)
pred = logreg.predict(X_test)
roc_auc_score(y_test, pred)
accuracy_score(y_test, pred)

logreg = LogisticRegression(penalty='none', max_iter=1e4)

# %% Fit base model with statsmodels
# mod = sm.discrete.discrete_model.Logit(y, X)
# fit = mod.fit()
# fit.summary()
# joblib.dump(mod, "02_77142-vcf_statsmodels-logistic-regression-model.pickle")
# fit.params
# %% Fit cluster model with statsmodels

# %% Fit principal component model with statsmodels

# %% Fit base model with scikit-learn
prefix = "02_77142-vcf_logistic-regression-model"

suffixes = [
    "age-gender-cluster-clade-region",
    "age-gender-cluster-clade",
    "age-gender-clade",
    "age-gender-cluster",
    "age-gender",
    "age",
]

X5 = X.loc[:, X.columns.str.contains("age|gen|clus|cla|reg")]
X4 = X.loc[:, X.columns.str.contains("age|gen|clus|cla")]
X3B = X.loc[:, X.columns.str.contains("age|gen|clad")]
X3A = X.loc[:, X.columns.str.contains("age|gen|clus")]
X2 = X[["covv_patient_age", "gender"]]
X1 = X[["covv_patient_age"]]

for x, s in zip([X5, X4, X3B, X3A, X2, X1], suffixes):
    logreg = LogisticRegression(penalty='none', max_iter=1e4)
    logreg.fit(x, y)
    joblib.dump(logreg, prefix + s + ".pickle")
    pred = logreg.predict(x)
    print(classification_report(y, pred))
    print(classification_report(y, pred))
    print(confusion_matrix(y, pred))
    precision_score(y, pred)
    recall_score(y, pred)
    f1_score(y, pred)
    pred = logreg.predict_proba(x)[::, 1]
    fpr, tpr, _ = roc_curve(y, pred)
    auc = roc_auc_score(y, pred)
    print(auc)
    plt.plot(fpr, tpr, label=f"{s.replace('-', ', ').title()}")
    plt.legend(loc=4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.tight_layout()
plt.savefig("plots/" + prefix + "_" + suffixes[0] + ".png")
plt.show()
X.shape
# %% Fit others model with scikit-learn
prefix = "02_77142-vcf_"

clfs = [
    DecisionTreeClassifier,
    KNeighborsClassifier,
    RandomForestClassifier
]
suffixes = [
    "dt",
    "knn",
    "rf",
]
for c, s in zip(clfs, suffixes):
    clf = c()
    clf.fit(X, y)
    joblib.dump(logreg, prefix + s + ".pickle")
    pred = logreg.predict(X)
    print(classification_report(y, pred))
    print(classification_report(y, pred))
    print(confusion_matrix(y, pred))
    precision_score(y, pred)
    recall_score(y, pred)
    f1_score(y, pred)
    pred = logreg.predict_proba(X)[::, 1]
    fpr, tpr, _ = roc_curve(y, pred)
    auc = roc_auc_score(y, pred)
    plt.plot(fpr, tpr, label=f"{s.upper()}, AUC={auc:.2f}")
    plt.legend(loc=4)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.tight_layout()
plt.savefig("plots/" + prefix + suffixes[-1] + ".png")
plt.show()
# %% Fit cluster model with scikit-learn

# %% Use sklearn logisitic regression model for prediction
pred = logreg.predict(X)

# %% Show model metrics

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
