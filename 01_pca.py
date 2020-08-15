import dask.dataframe as dd
import joblib
from sklearn.decomposition import PCA

ddf = dd.read_parquet("00_77142-vcf_wide.parquet").fillna(0)
pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
pca.fit(ddf.to_dask_array())
joblib.dump(pca, "01_77142-vcf_2-component-pca-model.pickle.gz")
