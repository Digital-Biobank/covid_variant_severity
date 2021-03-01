# %% Imports
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import chi2_contingency, norm
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    recall_score,
    precision_score,
    f1_score
)
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import MinMaxScaler

today = pd.Timestamp.today().date()

# %% Data
df = pd.read_parquet(f"data/2020-10-21_vcf-clean.parquet")

df = df.reset_index().drop_duplicates(subset="pid")

# %% Rescale age column
min_max_scaler = MinMaxScaler()
df["age"] = min_max_scaler.fit_transform(
    df["age"].values.reshape(-1, 1)
    )

# %% Drop missing values
df = df.dropna(subset=["age", "male"])
df = df.dropna(thresh=1, axis="columns")
df = df.fillna(0)

# %% Prepare data for logistic regression
y = df["is_red"]
X = df.drop([
    "is_red",
    'GH',
    'GR',
    'L',
    'O',
    'S',
    'V'
], axis=1)
X.assign(y=y).to_csv(f"data/2020-10-21_vcf-model.csv")

# %% Calculate variable frequency for plotting
var_freq = X.sum() / len(X)
var_freq.rename("variant_frequency").to_csv(
    f"data/2020-10-21_variant-freq.csv"
    )

# %% Fit logistic regression
lr = LogisticRegressionCV(penalty="l1", Cs=[1], cv=5, solver="liblinear")
lr.fit(X, y)
model = SelectFromModel(lr, prefit=True)
indices = model.get_support()
colnames = X.columns[indices]
colnames
lr.scores_[1].mean()
X_new = X.loc[:, indices]
X_new.assign(y=y).to_csv(
    f"data/2020-10-21_logistic-regression-lasso-selected-features.csv"
    )

coef_df = pd.DataFrame(lr.coef_, columns=X.columns)
ors = coef_df.squeeze().transform("exp")
ors = ors[ors != 1]
ors.sort_values().tail(20)
ors.to_csv(f"data/2020-10-21_odds-ratios.csv")

# Define functions need for Figure 2

def bootstrap_auc(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    auc_values = []
    for b in range(nsamples):
        idx = np.random.randint(X_train.shape[0], size=X_train.shape[0])
        clf.fit(X_train[idx], y_train[idx])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
        auc_values.append(roc_auc)
    return np.percentile(auc_values, (2.5, 97.5))

def permutation_test(clf, X_train, y_train, X_test, y_test, nsamples=1000):
    idx1 = np.arange(X_train.shape[0])
    idx2 = np.arange(X_test.shape[0])
    auc_values = np.empty(nsamples)
    for b in range(nsamples):
        np.random.shuffle(idx1)  # Shuffles in-place
        np.random.shuffle(idx2)
        clf.fit(X_train, y_train[idx1])
        pred = clf.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test[idx2].ravel(), pred.ravel())
        auc_values[b] = roc_auc
    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test.ravel(), pred.ravel())
    return roc_auc, np.mean(auc_values >= roc_auc)

def permutation_test_between_clfs(y_test, pred_proba_1, pred_proba_2, nsamples=1000):
    auc_differences = []
    auc1 = roc_auc_score(y_test.ravel(), pred_proba_1.ravel())
    auc2 = roc_auc_score(y_test.ravel(), pred_proba_2.ravel())
    observed_difference = auc1 - auc2
    for _ in range(nsamples):
        mask = np.random.randint(2, size=len(pred_proba_1.ravel()))
        p1 = np.where(mask, pred_proba_1.ravel(), pred_proba_2.ravel())
        p2 = np.where(mask, pred_proba_2.ravel(), pred_proba_1.ravel())
        auc1 = roc_auc_score(y_test.ravel(), p1)
        auc2 = roc_auc_score(y_test.ravel(), p2)
        auc_differences.append(auc1 - auc2)
    return observed_difference, np.mean(auc_differences >= observed_difference)

# %% Figure 2: Plot ROC curve
prefix = f"2020-10-21_vcf_logistic-regression-model"

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
clades = [
    'GH', 'GR', 'L', 'O', 'S', 'V'
]

continents = [
    "Asia", "Europe", "North America", "South America", "Oceania" 
]

X4 = df[["age", "male"] + continents + clades]
X3 = df[["age", "male"] + continents]
X2 = X[["age", "male"]]
X1 = X[["age"]]
for x, s, l in zip([X, X4, X3, X2, X1], suffixes, linestyles):
    X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.33,
        random_state=42
    )
    lr = LogisticRegressionCV(penalty="l1", Cs=[1], cv=5, solver="liblinear", random_state=1)
    # lr = LogisticRegression(penalty='none', solver="saga", max_iter=1e4, n_jobs=-1)
    lr.fit(X_train, y_train)
    joblib.dump(lr, f"models/{prefix}_{s}.pickle")
    pred = lr.predict(X_test)
    print("cross-val scores:", lr.scores_)
    print("cross-val score mean:", lr.scores_[1].mean())
    accuracy_score(y_test, pred)
    print(classification_report(y_test, pred))
    print(classification_report(y_test, pred))
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    odds_ratio = (tp*tn)/(fp*tn)
    se1 = np.sqrt(1/tn + 1/fp + 1/fn + 1/tp)
    top = odds_ratio + se1*1.96
    btm = odds_ratio - se1*1.96
    se2 = (top - btm)/(2*1.96)
    print("se1", se1, "se2", se2)
    z = np.log(odds_ratio) / se2
    p = np.exp(-.717*z - .416*z**2)
    sens = tp/(tp+fn)
    spec = tn/(tn+fp)
    neg_lr = (1 - sens) / spec
    print("negative likelihood ratio:", neg_lr)
    print(f"odds ratio: {odds_ratio:.1f} ({btm:.1f}-{top:.1f})")
    print(f"p-value: {p:.6f}")
    print(f"{s}:\nTrue negative: {tn}\nFalse positive: {fp}\nFalse negative: {fn}\nTrue positive: {tp}")
    precision_score(y_test, pred)
    recall_score(y_test, pred)
    f1_score(y_test, pred)
    pred = lr.predict_proba(X_test)[::, 1]
    fpr, tpr, _ = roc_curve(y_test, pred)
    auc = roc_auc_score(y_test, pred)
    print(auc)
    q1 = auc/(2-auc)
    q2 = 2*auc**2/(1+auc)
    auc_se = (auc*(1-auc)+(tp-1)*(q1-auc**2)+(tn-1)*(q2-auc**2))/(tn*tp)
    top = auc + auc_se*1.96
    btm = auc - auc_se*1.96
    print(f"AUC: {auc:.3f} ({btm:.4f}-{top:.4f})")
    print(f"AUC SE: {auc_se:.9f}")
    plt.plot(fpr, tpr, label=f"{s.replace('-', ', ').title()}", ls=l)
    plt.legend(loc=4)

agrv = joblib.load(f"models/{prefix}_{suffixes[0]}.pickle")

ag = joblib.load(f"models/{prefix}_{suffixes[3]}.pickle")
z = (0.9105254877281309 - 0.6792348437172226) / np.sqrt((0.000092628**2)+(0.017763603**2))
norm.sf(abs(z))*2 # 9.379754234884466e-39

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
plt.title("Mild/Severe Classification Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig(f"plots/{prefix}_{suffixes[0]}.png", dpi=300)
plt.show()


def plot_learning_curve(
    estimator,
    X,
    y,
    title="Learning Curve",
    axes=None,
    ylim=None,
    cv=None,
    n_jobs=None,
    train_sizes=np.linspace(.2, 1.0, 5),
    random_state=1
):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    random_state : int
        Required for reproducible results.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(10, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
        estimator,
        X,
        y,
        cv=cv,
        n_jobs=n_jobs,
        train_sizes=train_sizes,
        return_times=True,
        random_state=random_state
        )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="b")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="b",
                 label="Cross-validation score")
    axes.legend(loc="best")

    return plt


plot_learning_curve(
    LogisticRegression(penalty="l1", solver="liblinear"),
    X4,
    y,
    cv=5,
    random_state=1
)
plt.savefig(f"{today}_learning-curve.png")
plt.show()

df["pid"].to_csv(f"data/2020-10-21_pid.txt", index=False)

var_df = df.set_index("pid").iloc[:, :-13]
df.shape

p_list = [
    chi2_contingency(
        pd.crosstab(var_df["is_red"], var_df[feature])
    )[1]
    for feature in var_df.columns[1:]
    ]


p_list2 = [
    sm.stats.Table(
        pd.crosstab(var_df["is_red"], var_df[feature])
        ).test_ordinal_association().pvalue
    for feature in var_df.columns[1:]
    ]

t_list = [
    sm.stats.Table2x2(
        pd.crosstab(var_df["is_red"], var_df[feature])
        )
    for feature in var_df.columns[1:]
    ]

t_out = [(t.oddsratio, t.oddsratio_confint(), t.oddsratio_pvalue()) for t in t_list]
t_out[0]

pval_df = pd.DataFrame({
    "trend_test_pvalue": p_list2,
    "chi_square_pvalue": p_list
    },
    index=var_df.columns[1:]
    )

pval_df.to_csv(f"data/2020-10-21_p-values.csv")

# %%
df[df.iloc[:, 2:-13].sum(axis=1).gt(0)]
2853/3386