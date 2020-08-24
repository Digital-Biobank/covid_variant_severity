# %% Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans

# %% Read in PCA transformed data
df = pd.read_parquet("01_77142-vcf_2-component-pca-transformed.parquet")

# %% Read in outcome data and create label and patient ID (pid) columns
all_outcomes = pd.read_csv("200818_emm_IDsandstatus_all_plus.csv")
all_outcomes = all_outcomes.assign(
    all_labels=all_outcomes.covv_patient_status.map(
        {
            "Severe": 4,
            "Released_andor_Recovered": 3,
            "Live": 5,
            "Symptomatic": 1,
            "Mild": 2,
            "Deceased": 6,
            "Asymptomatic": 0,
        }
    ),
    is_red=all_outcomes.covv_patient_status.map(
        {
            "Asymptomatic": 0,
            "Deceased": 1,
            "Epidemiology Study": 0,
            "Live": 0,
            "Mild": 0,
            "Severe": 1,
            "Pneumonia (chest X-ray)": 1,
            "Pneumonia (chest X-ray), not critical": 1,
            "Screening": 1,
            "Symptomatic": 1,
        }
    ),
    # %% Binary encode mortality data and gender
    is_dead=all_outcomes["covv_patient_status"].map({"Live": 0, "Deceased": 1}),
    is_symp=all_outcomes["covv_patient_status"].map({"Asymptomatic": 0, "Symptomatic": 1}),
    is_sevr=all_outcomes["covv_patient_status"].map({"Mild": 0, "Severe": 1}),
    gender=all_outcomes["covv_gender"].map({"Female": 0, "Male": 1}),
    pid=all_outcomes["covv_accession_id"].str.extract(
        r"(\d+)",
        expand=False
    ).astype(int)
).set_index("pid")

# %% Join PCA transformed data with outcome data by patient ID (pid)
df = df.join(all_outcomes)

# %% Drop rows without one of the labels in all_labels
df = df.dropna(subset=["is_red"])

# %% Save combined PCA transformed and outcome data
df.to_parquet(
    "02_77142-vcf_2-component-pca-transformed_outcomes.parquet"
)

# %% Instantiate K-means model with three clusters
km = KMeans(n_clusters=3, random_state=42)

# %% Fit K-means model to principal components (PC1, PC2)
km.fit(df[["PC1", "PC2"]])

# %% Add clusters to combined PCA transformed and outcome data
df = df.assign(cluster=km.labels_)

# %% Save combined clusters, PCA transformed, and outcome data
df.to_parquet(
    "02_77142-vcf_"
    "2-component-pca-transformed_"
    "3-cluster-kmeans_"
    "outcomes"
    ".parquet"
)

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
    "01_77142-vcf_2-component-pca-components_top-variants_long.csv"
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
plt.savefig("02_77142-vcf_2-component-pca-_3-cluster-kmeans.png")

# %% Show plot
plt.show()
