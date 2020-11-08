# %% Imports
from operator import lt
import joblib
import pathlib
import statsmodels.api as sm
import numpy as np
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

df.to_csv(
    proj_dir / f"data/01_77142-vcf_wide_join_{TARGET_PART}.csv"
)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
# %% Convert categorical data into indicator (dummy) variables
X = pd.get_dummies(df, columns=["cat_region"], drop_first=True)
X = X.drop("region", axis=1)
X = X.drop("clade", axis=1)
# X = pd.get_dummies(X, columns=["clade"], drop_first=True)
X["covv_patient_age"] = min_max_scaler.fit_transform(X["covv_patient_age"].values.reshape(-1, 1))
X.head()
# %% Remove columns with low variance
y = X[TARGET_NAME]
X = X.drop([TARGET_NAME], axis=1)
lr = LogisticRegression(penalty="l1", solver="liblinear")
lr.fit(X, y)
coef_df = pd.DataFrame(lr.coef_, columns=X.columns)
ors = coef_df.squeeze().transform("exp")

ors.to_csv("data/odds_ratios.csv")

top_btm = ors[(ors > 2) | (ors < 0.5)].index
top_btm_X = X[top_btm]
top_btm_X.assign(y=y).to_csv("data/top_btm_df.csv")
ors.sort_values()
top_btm["odds_ratio"].min()

# %% Correlations
corr_mat = X.corr()
corr_mat.to_parquet("data/03_correlation-matrix_variants-region-sex-age.parquet")
corr_mat.to_csv("data/03_correlation-matrix_variants-region-sex-age.csv")
corr_mat.reset_index().to_feather("data/03_correlation-matrix_variants-region-clade-sex-age.feather")
# corr_mat = pd.read_parquet("data/03_correlation-matrix_variants-region-clade-sex-age.parquet")
# nz_corr_mat = corr_mat.loc[nz_coefs.index, nz_coefs.index]
# nz_corr_mat.shape
mask = np.tril(np.ones_like(corr_mat, dtype=bool))
corr_mat[mask] = np.nan
stack_corr = corr_mat.stack()
sort_corr = stack_corr.sort_values(kind="quicksort")
sort_corr.to_csv("data/03_sorted-correlation-matrix_variants-region-sex-age.csv")

sort_corr
# %% Use correlations to get list of variables to be removed
sort_corr = pd.read_csv("data/03_sorted-correlation-matrix_variants-region-sex-age.csv")
sort_corr.columns = ["var1", "var2", "corr"]
sort_corr = sort_corr.set_index(["var1", "var2"])

# %% Use correlations to get list of variables to be removed
gt95_0 = sort_corr[sort_corr["corr"] > .95].index.get_level_values(0).drop_duplicates()
gt95_1 = sort_corr[sort_corr["corr"] > .95].index.get_level_values(1).drop_duplicates()
gt90_0 = sort_corr[sort_corr["corr"] > .9].index.get_level_values(0).drop_duplicates()
gt90_1 = sort_corr[sort_corr["corr"] > .9].index.get_level_values(1).drop_duplicates()
gt85_0 = sort_corr[sort_corr["corr"] > .85].index.get_level_values(0).drop_duplicates()
gt85_1 = sort_corr[sort_corr["corr"] > .85].index.get_level_values(1).drop_duplicates()
gt80_0 = sort_corr[sort_corr["corr"] > .80].index.get_level_values(0).drop_duplicates()
gt80_1 = sort_corr[sort_corr["corr"] > .80].index.get_level_values(1).drop_duplicates()
gt70_0 = sort_corr[sort_corr["corr"] > .70].index.get_level_values(0).drop_duplicates()
gt60_0 = sort_corr[sort_corr["corr"] > .60].index.get_level_values(0).drop_duplicates()
gt50_0 = sort_corr[sort_corr["corr"] > .50].index.get_level_values(0).drop_duplicates()

# gt95_0.to_csv("data/correlated-variants.csv", index=False)
# gt95_0 = pd.read_csv("data/correlated-variants.csv")
# clade_corr.to_csv("data/clade-correlated-variants.csv")
# clade_corr = clade_corr.loc[:, ~clade_corr.columns.str.contains("clade")]
# stack_clade = clade_corr.stack().sort_values(kind="quicksort")
# stack_clade[stack_clade > .4]
# stack_clade.to_csv("data/stacked-clade-correlated-variants.csv")
# %% Drop highly correlated features
# Remove one of each highly correlated pair
lt95_0_X = X.drop(gt95_0.values, axis=1)
lt90_0_X = X.drop(gt90_0.values, axis=1)
lt85_0_X = X.drop(gt85_0.values, axis=1)
lt80_0_X = X.drop(gt80_0.values, axis=1)
lt70_0_X = X.drop(gt70_0.values, axis=1)
lt60_0_X = X.drop(gt60_0.values, axis=1)
lt50_0_X = X.drop(gt50_0.values, axis=1)

# %% Train-test split
hv_X_train, hv_X_test, hv_y_train, hv_y_test = train_test_split(
    hv_X,
    y,
    test_size=0.33,
    random_state=42
)

cols = pd.read_csv(
    "03_red-target_logistic-regression_top-and-bottom-odds-ratios.csv",
    index_col=0
    )
X
top_btm = X[cols.squeeze().tolist() + ["covv_patient_age", "gender"]]
corr_mat = top_btm.corr()
mask = np.tril(np.ones_like(corr_mat, dtype=bool))
corr_mat[mask] = np.nan
stack_corr = corr_mat.stack()
sort_corr = stack_corr.sort_values(kind="quicksort")
sort_corr.to_csv("data/03_red-target_sorted-correlation-matrix_top-and-bottom-odds-ratio.csv")
sort_corr
# %% Use correlations to get list of variables to be removed
sort_corr = pd.read_csv("data/03_red-target_sorted-correlation-matrix_top-and-bottom-odds-ratio.csv")
sort_corr.columns = ["var1", "var2", "corr"]
sort_corr = sort_corr.set_index(["var1", "var2"])
gt95_0 = sort_corr[sort_corr["corr"] > .0001].index.get_level_values(0).drop_duplicates()
lt95_0_X = top_btm.drop(gt95_0.values, axis=1)
corr_mat = lt.corr()
mask = np.tril(np.ones_like(corr_mat, dtype=bool))
corr_mat[mask] = np.nan
lt95_0_X.corr().stack().sort_values(kind="quicksort")
lt95_0_X
hv_X = lt95_0_X.loc[:, lt95_0_X.var() > 0.01]
hv_X
X_train, X_test, y_train, y_test = train_test_split(
    top_btm.iloc[:,:9],
    y,
    test_size=0.33,
    random_state=42
)
# %% Fit statsmodels logistic regression

lr = sm.Logit(y_train, X_train)
fit = lr.fit(maxiter=1000)
joblib.dump(
    fit,
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
                          "correlation-cutoff-80_statsmodels.pickle"
)
fit.summary()
dir(fit)
ci = fit.conf_int()
ci
# fit

# %% Read in non-zero coefficients
nz_coefs = pd.read_csv(
    "03_red-target_logistic-regression-coefs_non-zero-coefficients.csv",
    index_col=0
)
# %% Remove non-zero coefficients
nz_vars = nz_coefs.index.tolist()
nz_v = [v for v in nz_coefs if v in lt80_0_X.columns]
nz_vars
# %% Remove non-zero coefficients
nz_lt80_0_X = lt80_0_X[nz_v]

# TODO scikit-learn: 5 curves on ROC plot, 2 plots one for each level (0, 1)
# TODO statsmodels: pick lowest acceptable threshold and fit model
# TODO statsmodels: use fit model to plot p-value plots and run LRT

# lt95_X.to_parquet("data/03_95-correlation-threshold_variants-region-sex-age-dataframe.parquet")
# lt95_X = pd.read_parquet("data/03_95-correlation-threshold_variants-region-sex-age-dataframe.parquet")
N = 20
top_corrs = not_one_corr.drop(not_one_corr.index[N//2:-N//2])
top_corrs = not_one_corr.drop(not_one_corr.index[-N:])
top_corrs.to_csv("data/03_top-correlations_variants-region-sex-age.csv")
top_corrs = pd.read_csv("data/03_top-correlations_variants-region-sex-age.csv")
top_corrs
my_norm = Normalize(vmin=top_corrs.min(), vmax=top_corrs.max())
cmap = plt.cm.get_cmap('coolwarm', N)
top_corrs.plot.barh(
    legend=False,
    color=cmap(my_norm(top_corrs)),
    edgecolor="k"
)
plt.tight_layout()
plt.title(f"Most Correlated Variables")
plt.xlabel("Correlation")
plt.ylabel("Variant")
plt.savefig(f"plots/top_{TARGET_PART}_variable_corrs.png")
plt.show()
import seaborn as sns
sns.distplot(
    not_one_corr, rug=True, axlabel="Correlation Frequency",
    rug_kws={"color": "k", "height": .02},
    kde_kws={"color": "b", "lw": 2},
    hist=False
)
plt.title("Distribution of correlations")
plt.ylabel("Density")
plt.tight_layout()
plt.savefig("plots/00_77142-vcf_variant-correlation-kde.png")
plt.show()


# %% Prepare scikit-learn data
y = X[TARGET_NAME]
X = X.drop([TARGET_NAME], axis=1)

# %% Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42
)


# %% Instantiate, fit, and assess scikit-learn model
correg = LogisticRegression(
    solver="saga",
    max_iter=1e4,
    random_state=42,
    n_jobs=-1
)

correg.fit(lt95_X_train, lt95_y_train)
pred = correg.predict(lt95_X_test)
roc_auc_score(lt95_y_test, pred)
accuracy_score(lt95_y_test, pred)

# %% Save scikit-learn model
joblib.dump(
    correg,
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
    "correlation-cutoff-95.pickle"
)

# %% Load scikit-learn model
correg = joblib.load(
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
    "correlation-cutoff-95.pickle"
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

# %% Save scikit-learn model
joblib.dump(
    logreg,
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
    "l1-penalty_5-fold-cv_100-Cs_66-percent-train-size.pickle"
)

# %% Load scikit-learn model
logreg = joblib.load(
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
    "l1-penalty_5-fold-cv_100-Cs_66-percent-train-size.pickle"
)

# %% Compare LASSO and regular Logistic Regression
l1_reg = LogisticRegression(
    penalty='l1',
    solver="saga",
    C=logreg.C_[0],
    max_iter=1e4,
    n_jobs=-1
)

l1_reg.fit(X_train, y_train)
pred = l1_reg.predict(X_test)
roc_auc_score(y_test, pred)
accuracy_score(y_test, pred)

# %% Save scikit-learn model
joblib.dump(
    l1_reg,
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
    "l1-penalty_66-percent-train-size.pickle"
)

no_reg = LogisticRegression(
    penalty='none',
    solver="saga",
    max_iter=1e4,
    n_jobs=-1
)

no_reg.fit(X_train, y_train)
pred = no_reg.predict(X_test)
roc_auc_score(y_test, pred)
accuracy_score(y_test, pred)

# %% Save scikit-learn model
joblib.dump(
    no_reg,
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
    "no-penalty_66-percent-train-size.pickle"
)

# %% Load scikit-learn model
l1_reg = joblib.load(
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
                          "l1-penalty_66-percent-train-size.pickle"
)

no_reg = joblib.load(
    proj_dir / "models" / "03_red-target_logistic-regression-model_"
                          "no-penalty_66-percent-train-size.pickle"
)

model_names = "n=23671", "n=20982"
pred = no_reg.predict(X_test)
roc_auc_score(y_test, pred)
accuracy_score(y_test, pred)
print(classification_report(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
precision_score(y_test, pred)
recall_score(y_test, pred)
f1_score(y_test, pred)
pred = no_reg.predict_proba(X_test)[::, 1]
fpr, tpr, _ = roc_curve(y_test, pred)
auc = roc_auc_score(y_test, pred)
print(auc)
plt.plot(fpr, tpr, label=f"{model_names[0].replace('-', ', ').title()}")
plt.legend(loc=4)

pred = correg.predict(lt95_X_test)
roc_auc_score(lt95_y_test, pred)
accuracy_score(lt95_y_test, pred)
print(classification_report(lt95_y_test, pred))
print(classification_report(lt95_y_test, pred))
print(confusion_matrix(lt95_y_test, pred))
precision_score(lt95_y_test, pred)
recall_score(lt95_y_test, pred)
f1_score(lt95_y_test, pred)
pred = correg.predict_proba(lt95_X_test)[::, 1]
fpr, tpr, _ = roc_curve(lt95_y_test, pred)
auc = roc_auc_score(lt95_y_test, pred)
print(auc)
plt.plot(fpr, tpr, label=f"{model_names[1].replace('-', ', ').title()}")
plt.legend(loc=4)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.tight_layout()
plt.title(f"{TARGET_FULL} Classification Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("plots/" + "corr-thres_vs_no-corr-thres.png")
plt.show()

# %% Plot sklearn coefficients
coefs = pd.DataFrame(no_reg.coef_, columns=X.columns).squeeze()
nz_coefs = coefs[coefs != 0]
nz_coefs.shape
sorted_coefs = nz_coefs.sort_values()
sorted_coefs.index[0]
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
gt95.shape
lt95_X.shape
# %% Save sklearn coefficients
nz_coefs.to_csv(
    "03_red-target_logistic-regression-coefs_non-zero-coefficients.csv"
)

# %% Load sklearn coefficients
nz_coefs = pd.read_csv(
    "03_red-target_logistic-regression-coefs_non-zero-coefficients.csv",
    index_col=0
)

# %% Compare all variants and non-zero coef variants Logistic Regression
nz_reg = LogisticRegression(
    solver="saga",
    max_iter=1e4,
    n_jobs=-1
)

# %% Train model with nz_coefficients only
indices = np.array([c in nz_coefs.index for c in coefs.index])
nz_X_train = X_train.loc[:, indices]
nz_X_test = X_test.loc[:, indices]
nz_reg.fit(nz_X_train, y_train)
pred = nz_reg.predict(nz_X_test)
roc_auc_score(y_test, pred)
accuracy_score(y_test, pred)


# %% Plot ROC Curve
model_names = "No Penalty (23677 columns)"#, "L1 Penalty (23677 columns)"
for m, s in zip([no_reg], model_names):
    pred = m.predict(X_test)
    roc_auc_score(y_test, pred)
    accuracy_score(y_test, pred)
    print(classification_report(y_test, pred))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    precision_score(y_test, pred)
    recall_score(y_test, pred)
    f1_score(y_test, pred)
    pred = m.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    print(auc)
    plt.plot(fpr, tpr, label=f"{s.replace('-', ', ').title()}")
    plt.legend(loc=4)

s = "No Penalty (4219 columns)"
m = nz_reg
pred = m.predict(X_test.loc[:, indices])
roc_auc_score(y_test, pred)
accuracy_score(y_test, pred)
print(classification_report(y_test, pred))
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))
precision_score(y_test, pred)
recall_score(y_test, pred)
f1_score(y_test, pred)
pred = m.predict_proba(X_test.loc[:, indices])[::, 1]
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
plt.savefig("plots/" + "l1_vs_no-penalty_feature-selection.png")
plt.show()

# %% Plot statsmodels coefficients

nz_coefs.index in nz_coefs.index
nzX = lt95_X[nz_coefs.index]
lr = sm.Logit(y, nzX)
fit = lr.fit(maxiter=1000)
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


# %% Figure 1: Plot ROC curve
prefix = "03_77142-vcf_logistic-regression-model"

suffixes = [
    "age-gender-region-variant",
    "age-gender-region-clade",
    "age-gender-region",
    "age-gender",
    "age",
]

linestyles = [
    "-",
    "--",
    "-.",
    ":",
    "-",
]

X5 = X
X4 = X.loc[:, X.columns.str.contains("age|gen|reg|clade")]
X3 = X.loc[:, X.columns.str.contains("age|gen|reg")]
X2 = X[["covv_patient_age", "gender"]]
X1 = X[["covv_patient_age"]]
for x, s, l in zip([X5, X4, X3, X2, X1], suffixes, linestyles):
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        random_state=42
    )
    lr = LogisticRegression(penalty='l1', solver="liblinear", max_iter=1e4)
    # lr = LogisticRegression(penalty='none', solver="saga", max_iter=1e4, n_jobs=-1)
    lr.fit(X_train, y_train)
    joblib.dump(lr, prefix + s + ".pickle")
    pred = lr.predict(X_test)
    roc_auc_score(y_test, pred)
    accuracy_score(y_test, pred)
    print(classification_report(y_test, pred))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    precision_score(y_test, pred)
    recall_score(y_test, pred)
    f1_score(y_test, pred)
    pred = lr.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    print(auc)
    plt.plot(fpr, tpr, label=f"{s.replace('-', ', ').title()}", ls=l)
    plt.legend(loc=4)

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.title(f"{TARGET_FULL} Classification Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig("plots/" + prefix + "_" + suffixes[0] + ".png", dpi=300)
plt.show()

# %% Plot ROC curve
prefix = "03_77142-vcf_logistic-regression-model"

suffixes = [
    "no threshold",
    "95% threshold",
    "90% threshold",
    "85% threshold",
    "80% threshold",
]

for x, s in zip([X, lt95_0_X, lt90_0_X, lt85_0_X, lt80_0_X], suffixes):
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        random_state=42
    )
    lr = LogisticRegression(penalty='none', solver="saga", max_iter=1e4, n_jobs=-1)
    lr.fit(X_train, y_train)
    joblib.dump(lr, prefix + s + ".pickle")
    pred = lr.predict(X_test)
    roc_auc_score(y_test, pred)
    accuracy_score(y_test, pred)
    print(classification_report(y_test, pred))
    print(classification_report(y_test, pred))
    print(confusion_matrix(y_test, pred))
    precision_score(y_test, pred)
    recall_score(y_test, pred)
    f1_score(y_test, pred)
    pred = lr.predict_proba(X_test)[::, 1]
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
