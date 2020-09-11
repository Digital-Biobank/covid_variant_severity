# %% Imports
import joblib
import pathlib
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, \
    f1_score
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# %% Variables
proj_dir = pathlib.Path.home() / "covid" / "vcf"

TARGET_NAME = "is_red"
TARGET_PART = "red"
TARGET_FULL = "Green/Red"

# %% Data
df = pd.read_parquet(
    proj_dir / f"data/01_77142-vcf_wide_join_{TARGET_PART}.parquet"
)

# %% Convert categorical data into indicator (dummy) variables
X = pd.get_dummies(df, columns=["cat_region", "clade"], drop_first=True)
X = X.drop("region", axis=1)
# %% Prepare scikit-learn data
y = X[TARGET_NAME]
X = X.drop([TARGET_NAME], axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)

# %% Instantiate, fit, and assess scikit-learn model
logreg = LogisticRegressionCV(
    penalty='l1',
    Cs=100,
    solver="saga",
    max_iter=1e4,
    random_state=42,
    n_jobs=-1
)

logreg.fit(X_train, y_train)
pred = logreg.predict(X_test)
roc_auc_score(y_test, pred)
accuracy_score(y_test, pred)

joblib.dump(
    logreg,
    "03_red-target_logistic-regression-model_"
    "l1-penalty_5-fold-cv_100-Cs_66-percent-train-size.pickle"
)

# %% Plot sklearn coefficients
coefs = pd.DataFrame(logreg.coef_, columns=X.columns).squeeze()
nz_coefs = coefs[coefs != 0]
sorted_coefs = nz_coefs.sort_values()
sorted_coefs.index = ["33bp del at 499"] + list(sorted_coefs.index[1:])
N = 20
cmap = plt.cm.get_cmap('coolwarm', N)
top_coefs = sorted_coefs.drop(sorted_coefs.index[N//2:-N//2])
my_norm = Normalize(vmin=top_coefs.min(), vmax=top_coefs.max())
top_coefs.plot.barh(
    legend=False,
    color=cmap(my_norm(top_coefs)),
    edgecolor="k"

)
plt.tight_layout()
plt.title(f"Top {TARGET_FULL} Logistic Regression Variables")
plt.xlabel("Coefficient")
plt.ylabel("Variant")
plt.savefig(f"top_{TARGET_PART}_variable_coefs.png")
plt.show()

# %% Save sklearn coefficients
nz_coefs.to_csv(
    "03_red-target_logistic-regression-coefs_"
    "l1-penalty_5-fold-cv_100-Cs_66-percent-train-size.pickle"
)

nz_coefs = pd.read_csv(
    "03_red-target_logistic-regression-coefs_"
    "l1-penalty_5-fold-cv_100-Cs_66-percent-train-size.pickle"
)

# %% Plot statsmodels coefficients
nzX = X[nz_coefs.index]
lr = sm.Logit(y, nzX)
fit = lr.fit_regularized(alpha=1/logreg.C_)
fit.summary()
err_series = fit.params - fit.conf_int()[0]

# %% Save statsmodels logistic regression model
joblib.dump(
    fit,
    "03_red-target_logistic-regression-model_"
    "l1-penalty_statsmodels.pickle"
)

coef_df = pd.DataFrame({'coef': fit.params.values[1:],
                        'err': err_series.values[1:],
                        'varname': err_series.index.values[1:]
                       })

coef_df.to_csv(
    "03_red-target_statsmodels-logistic-regression-model.pickle"
)

fig, ax = plt.subplots(figsize=(8, 5))
coef_df.plot.barh(
    x='varname',
    y='coef',
    ax=ax,
    color='none',
    yerr='err',
    legend=False
)
ax.set_ylabel('')
ax.set_xlabel('')
ax.scatter(
    x=coef_df['coef'],
    y=pd.np.arange(coef_df.shape[0]),
    marker='s',
    s=120,
    color='black'
)
ax.axvline(x=0, linestyle='--', color='black', linewidth=4)
ax.xaxis.set_ticks_position('none')
_ = ax.set_yticklabels(coef_df["varname"], rotation=0, fontsize=16)
plt.savefig(
    "03_red_statsmodels-logistic-regression-coefplot.png"
)
plt.show()


# %% Plot ROC curve
prefix = "03_77142-vcf_logistic-regression-model"

suffixes = [
    "age-gender-region-clade-variant",
    "age-gender-region-clade",
    "age-gender-region",
    "age-gender",
    "age",
]
X4 = X.loc[:, X.columns.str.contains("age|gen|reg|clade")]
X3 = X.loc[:, X.columns.str.contains("age|gen|reg")]
X2 = X[["covv_patient_age", "gender"]]
X1 = X[["covv_patient_age"]]

for x, s in zip([X, X4, X3, X2, X1], suffixes):
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        random_state=42
    )
    lr = LogisticRegression(penalty='l1', solver="saga", C=logreg.C_[0], max_iter=1e4)
    lr.fit(X_train, y_train)
    joblib.dump(logreg, prefix + s + ".pickle")
    pred = logreg.predict(X_test)
    roc_auc_score(y_test, pred)
    accuracy_score(y_test, pred)
    print(classification_report(y_test, pred))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    precision_score(y_test, pred)
    recall_score(y_test, pred)
    f1_score(y_test, pred)
    pred = logreg.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    print(auc)
    plt.plot(fpr, tpr, label=f"{s.replace('-', ', ').title()}")
    plt.legend(loc=4)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.tight_layout()
plt.title(f"{TARGET_FULL} Classification Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("plots/" + prefix + "_" + suffixes[0] + ".png")
plt.show()

# %% Semi-supervised learning
# %% Prepare data for scikit-learn
df = pd.read_parquet(proj_dir / "data/02_semi_knn-kernel-label-spreading-dataframe.parquet")
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

cols = [
    "is_red",
    "covv_patient_age",
    "gender",
    "clade",
    "cat_region",
]  # + ["PC" + str(i) for i in range(1, 10)]
cols
df.columns
df = df[cols].dropna().astype({
    "is_red": int,
    "gender": int,
    "clade": int,
    "cat_region": int,
})
true_X = df.dropna()
true_y = true_X["is_red"]
true_X = true_X.drop(["is_red"], axis=1)
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

# %% Fit base model with statsmodels
# mod = sm.discrete.discrete_model.Logit(y, X)
# fit = mod.fit()
# fit.summary()
# joblib.dump(mod, "02_77142-vcf_statsmodels-logistic-regression-model.pickle")
# fit.params
# %% Fit cluster model with statsmodels

# %% Fit principal component model with statsmodels


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
