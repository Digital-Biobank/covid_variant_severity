import joblib
import dask.dataframe as dd


ddf = dd.read_parquet("00_77142-vcf_wide.parquet").fillna(0)
pca = joblib.load("01_77142-vcf_2-component-pca-model.pickle.gz")
X = pca.transform(ddf.to_dask_array())
Xdf = dd.from_array(X)
Xdf.to_parquet("02_77142-vcf_2-component-pca-transformed-data.parquet")
