# %% Imports
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# %% Read in VCF wide data
df = pd.read_parquet("data/00_77142-vcf_wide.parquet").fillna(0)

# %% Instantiate PCA model
pca = PCA(n_components=9, svd_solver="randomized", random_state=42)

pc_cols = ["PC" + str(i) for i in range(1, 10)]

# %% Fit PCA model and create transformed data for plotting
transformed = pd.DataFrame(
    pca.fit_transform(df),
    # Add patient IDs (pid) index to transformed data
    index=df.index,
    # Label transformed data columns as PC1 and PC2
    columns=pc_cols
)

# %% Save transformed data for plotting
transformed.reset_index().to_feather(
    "data/01_77142-vcf_9-component-pca-transformed.feather"
)
transformed.to_parquet(
    "data/01_77142-vcf_9-component-pca-transformed.parquet"
)

# %% Save PCA model
joblib.dump(pca, "models/01_77142-vcf_9-component-pca-model.pickle.gz")

# %% Put PCA components (loadings) into a dataframe
pca_df = pd.DataFrame({
    "PC1": pca.components_[0],
    "PC2": pca.components_[1],
    "PC3": pca.components_[2],
    "PC4": pca.components_[3],
    "PC5": pca.components_[4],
    "PC6": pca.components_[5],
    "PC7": pca.components_[6],
    "PC8": pca.components_[7],
    "PC9": pca.components_[8],
    },
    index=df.columns
)

# %% Save PCA components (loadings)
pca_df.reset_index().to_feather(
    "data/01_77142-vcf_9-component-pca-components.feather"
)
pca_df.to_parquet(
    "data/01_77142-vcf_9-component-pca-components.parquet"
)

# %% List variants with highest PCA component correlations
variants = {
    # PC1 low
    "A23403G",
    "C3037T",
    "C14408T",
    "C241T",
    # PC1 low, PC2 high
    "G28882A",
    "G28883C",
    "G28881A",
    # PC1 high
    "G26144T",
    "C8782T",
    "T28144C",
    "C14805T",
    "G11083T",
    # PC2 low
    "G25563T",
    "C1059T",
    "C241T",
    "C3037T",
    "C14408T",
    "A23403G",
}

# %% Filter data other variants
pca_plot_data = pca_df.loc[variants]

# %% Save top variant PCA component correlations
pca_plot_data.T.to_csv(
    "data/01_77142-vcf_2-component-pca-components_top-variants_wide.csv"
)
pca_plot_data.to_csv(
    "data/01_77142-vcf_2-component-pca-components_top-variants_long.csv"
)

# %% Read in outcome data and create label and patient ID (pid) columns
df2 = pd.read_csv("data/2020-09-01all_cleaned_GISAID0901pull.csv")
df2 = df2.assign(
    pid=df2["covv_accession_id"].str.extract(
        r"(\d+)",
        expand=False
    ).astype(int),
    covv_patient_status=df2["covv_patient_status"].str.strip(),
    # %% Binary encode mortality data and gender
    is_dead=df2["covv_patient_status"].map({"Live": 0, "Deceased": 1}),
    is_symp=df2["covv_patient_status"].map({"Asymptomatic": 0, "Symptomatic": 1}),
    is_sevr=df2["covv_patient_status"].map({"Mild": 0, "Severe": 1}),
    gender=df2["covv_gender"].map({"Female": 0, "Male": 1, "Kvinna": 0}),
    covv_clade=df2["covv_clade"].astype("str")
).set_index("pid")
region_key = enumerate(sorted(df2["region"].unique()))
clade_key = enumerate(sorted(df2["covv_clade"].unique()))
df2 = df2.assign(
    is_red=df2["covv_patient_status"].map(
        {
            # Green
            "Asymptomatic": 0,
            "Released": 0,
            "Recovered": 0,
            "Live": 0,
            "Mild": 0,
            # Red
            "Deceased": 1,
            "Severe": 1,
            "Symptomatic": 1,
        }
    ),
    cat_region=df2["region"].map({v: k for k, v in region_key}),
    clade=df2["covv_clade"].map({v: k for k, v in clade_key}),
)
region_key = enumerate(sorted(df2["region"].unique()))
list(region_key)

# %% Join PCA transformed data with outcome data by patient ID (pid)
df = transformed.join(df2)

# %% Drop rows without one of the labels in all_labels
df = df.dropna(subset=["is_red"])

# %% Save combined PCA transformed and outcome data
df.reset_index().to_feather(
    "data/02_77142-vcf_9-component-pca-transformed_red.parquet"
)
df.to_parquet(
    "data/02_77142-vcf_9-component-pca-transformed_red.parquet"
)

# %% Instantiate K-means model with three clusters
km = KMeans(n_clusters=3, random_state=42)

# %% Fit K-means model to principal components (PC1, PC2)
km.fit(df[pc_cols])

# %% Add clusters to combined PCA transformed and outcome data
df = df.assign(cluster=km.labels_)

# %% Save combined clusters, PCA transformed, and outcome data
df.to_parquet(
    "data/02_77142-vcf_9-component-pca-transformed_"
    "3-cluster-kmeans.parquet"
)
df.columns
df.columns
# %% Drop rows without one of the labels in is_red
red = df.dropna(subset=["is_red"])
dead = df.dropna(subset=["is_dead"])
symp = df.dropna(subset=["is_symp"])
sevr = df.dropna(subset=["is_sevr"])

# %% Compare clusters
compare_df = df.groupby(["cluster", "covv_patient_status"]).size().unstack()
tdf = compare_df.T
relative_df = tdf / tdf.sum()
relative_df.columns = ["red", "green", "blue"]
relative_df[["red", "green", "blue"]].divide(relative_df["green"], axis=0)

# %% Plot clusters using seaborn
# sns.scatterplot(
#     x="PC1",
#     y="PC2",
#     alpha=0.2,
#     s=256,
#     hue="cluster",
#     data=dead
# )
# plt.show()

# %% Read in top variants for variable projection plotting using matplotlib
top_vars = pd.read_csv(
    "data/01_77142-vcf_2-component-pca-components_top-variants_long.csv"
)

# %% Plot clusters and variable projections using matplotlib
colormap = np.array(['b', 'g', 'r'])
plt.scatter(
    x=red.iloc[:, 0],
    y=red.iloc[:, 1],
    s=256,
    c=colormap[red["cluster"]],
    alpha=0.4
)

# %% Plot clusters and variable projections using matplotlib

# %% Use quiver to generate the variable projections
plt.quiver(
    np.zeros(top_vars.shape[0]),
    np.zeros(top_vars.shape[0]),
    top_vars.iloc[:, 1],
    top_vars.iloc[:, 2],
    angles='xy',
    scale_units='xy',
    scale=1
)

# %% Add unit circle to show perfect correlation between variants and components
# circle = plt.Circle(
#     (0, 0),
#     1,
#     facecolor='none',
#     edgecolor='b'
# )
# plt.gca().add_artist(circle)

# %% Save plot
plt.tight_layout()
plt.savefig("plots/02_77142-vcf_2-component-pca-_3-cluster-kmeans.png")

# %% Show plot
plt.show()
