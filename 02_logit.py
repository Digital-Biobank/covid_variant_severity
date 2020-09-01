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
from sklearn.metrics import roc_curve, roc_auc_score

# %% Read in combined clusters, PCA transformed, and mortality data
df = pd.read_csv("data/01_77142-vcf_wide_mutations_red.csv")

# %% Prepare data for scikit-learn
X = df[[
    "is_red",
    "gender",
    "clade",
    "cat_region",
    "covv_patient_age",
    "mutation_count",
]].dropna()
y = X["is_red"]
X = X.drop("is_red", axis=1)

# %% Convert categorical data into indicator (dummy) variables
X = pd.get_dummies(X, columns=["cat_region", "clade"], drop_first=True)

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
    "age-gender-mutation-clade-region",
    "age-gender-mutation-clade",
    "age-gender-mutation",
    "age-gender",
    "age",
]

X1 = X[["covv_patient_age"]]
X2 = X[["covv_patient_age", "gender"]]
X3 = X[["covv_patient_age", "gender", "mutation_count"]]
X4 = X.loc[:, X.columns.str.contains("age|gen|mut|cla")]

for x, s in zip([X, X4, X3, X2, X1], suffixes):
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
plt.savefig("plots/" + prefix + suffixes[-1] + ".png")
plt.show()

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
