import joblib
import numpy as np
import pandas as pd
from sklearn.semi_supervised import LabelSpreading

df = pd.read_parquet("data/01_77142-vcf_wide_mutations_red.parquet")
X = df.drop([
    "is_dead",
    "is_symp",
    "is_sevr",
    "sequence_length"
], axis=1).select_dtypes("number")
X["is_red"] = X["is_red"].fillna(-1)
X = X.dropna()
y = X["is_red"]
X = X.drop("is_red", axis=1)
cols = X.columns
rows = X.index
label_spread = LabelSpreading(kernel='knn', alpha=0.8, max_iter=1000)
label_spread.fit(X, y)
output_labels = label_spread.transduction_
joblib.dump(label_spread, "models/02_semi_knn-kernel-label-spreading-model.pickle")
df = X.assign(is_red=y, spread_labels=output_labels)
df["is_red"] = df["is_red"].replace(-1, np.nan)
df["is_red"].isna().sum()
df.to_parquet("data/02_semi_knn-kernel-label-spreading-dataframe.parquet")
