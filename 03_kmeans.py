import joblib
import dask.dataframe as dd
from dask_ml.cluster import KMeans
import matplotlib.pyplot as plt

Xdf = dd.read_parquet(
    "02_77142-vcf_2-component-pca-transformed-data.parquet"
)


Xdf.to_csv("02_77142-vcf_2-component-pca-transformed-data.csv")
km = KMeans(n_clusters=3)
km.fit(Xdf)
joblib.dump(Xdf, "02_77142-vcf_2-component-pca-transformed-data.pickle.gz")
joblib.dump(km, "03_77142-vcf_2-component-pca_kmeans-model.pickle.gz")
joblib.dump(km, "03_77142-vcf_2-component-pca_kmeans-model.pickle.gz")
labs = km.labels_.compute()
x0 = Xdf[0].compute()
x1 = Xdf[1].compute()
plt.scatter(x=x0, y=x1, c=labs)
plt.show()
