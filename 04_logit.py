# %% Imports
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import classification_report, confusion_matrix, recall_score, precision_score, \
    f1_score
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
import statsmodels.api as sm
from scipy.stats import chi2_contingency


# %% Data
df = pd.read_parquet("data/2020-10-21_vcf-clean.parquet")

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
X.assign(y=y).to_csv("data/2020-10-21_vcf-model.csv")

# %% Calculate variable frequency for plotting
var_freq = X.sum() / len(X)
var_freq.rename("variant_frequency").to_csv("data/variant-freq.csv")

# %% Fit logistic regression
lr = LogisticRegression(penalty="l1", solver="liblinear")
lr.fit(X, y)
model = SelectFromModel(lr, prefit=True)
indices = model.get_support()
colnames = X.columns[indices]
colnames
X_new = X.loc[:, indices]
X_new.assign(y=y).to_csv(
    "data/2020-10-21_logistic-regression-lasso-selected-features.csv"
    )

coef_df = pd.DataFrame(lr.coef_, columns=X.columns)
ors = coef_df.squeeze().transform("exp")
ors = ors[ors != 1]
ors.sort_values().tail(20)
ors.to_csv("data/2020-10-21_odds-ratios.csv")

# %% Figure 2: Plot ROC curve
prefix = "2020-10-21_vcf_logistic-regression-model"

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
    lr = LogisticRegression(penalty='l1', solver="liblinear", max_iter=1e4)
    # lr = LogisticRegression(penalty='none', solver="saga", max_iter=1e4, n_jobs=-1)
    lr.fit(X_train, y_train)
    joblib.dump(lr, prefix + s + ".pickle")
    pred = lr.predict(X_test)
    roc_auc_score(y_test, pred)
    accuracy_score(y_test, pred)
    print(classification_report(y_test, pred))
    print(classification_report(y_test, pred))
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    print(f"{s}:\nTrue negative: {tn}\nFalse positive: {fp}\nFalse negative: {fn}\nTrue positive: {tp}")
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
plt.title("Green/Red Classification Logistic Regression")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.tight_layout()
plt.savefig("plots/" + prefix + "_" + suffixes[0] + ".png", dpi=300)
plt.show()

df["pid"].to_csv("data/2020-10-21_pid.txt", index=False)

var_df = df.set_index("pid").iloc[:, :-13]

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

pval_df = pd.DataFrame({
    "trend_test_pvalue": p_list2,
    "chi_square_pvalue": p_list
    },
    index=var_df.columns[1:]
    )
# pval_df["POS"] = pval_df.index.str.replace(r"\D", "")

pval_df.to_csv(f"data/2020-10-21_p-values.csv")