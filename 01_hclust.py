# %% Imports
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram

# %% Read in VCF wide data
df = pd.read_parquet("data/00_77142-vcf_wide.parquet").fillna(0)

# %% Instantiate AggClust model
ac = AgglomerativeClustering()

clusters = ac.fit_predict(df)

# %% Fit PCA model and create transformed data for plotting

# %% Save hclust model
joblib.dump(ac, "models/01_77142-vcf_2-cluster-hclust-model.pickle")
joblib.dump(clusters, "models/01_77142-vcf_2-cluster-hclust-clusters.pickle")

df = df.assign(cluster=clusters)
# %% Save transformed data for plotting
df.reset_index().to_feather(
    "data/01_77142-vcf_2-component-hclust-clusters.feather"
)
df.to_parquet("data/01_77142-vcf_2-component-hclust-clusters.parquet")
joblib.dump(df, "data/01_77142-vcf_2-component-hclust-clusters.parquet")

# %% Read in outcome data and create label and patient ID (pid) columns
all_outcomes = pd.read_csv("data/2020-08-25_cleaned_GISAID.csv")
all_outcomes = all_outcomes.assign(
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
df.reset_index().to_feather(
    "data/02_77142-vcf_2-component-hclust-clusters_outcomes.feather"
)
df.to_parquet(
    "data/02_77142-vcf_2-component-hclust-clusters_outcomes.parquet"
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

def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_,
                                      counts]).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)


plt.title('Hierarchical Clustering Dendrogram')
# plot the top three levels of the dendrogram
plot_dendrogram(ac, truncate_mode='level', p=3)
plt.xlabel("Number of points in node (or index of point if no parenthesis).")

# %% Save plot
plt.tight_layout()
plt.savefig("plots/02_77142-vcf_2-component-hclust-dendrogram.png")

# %% Show plot
plt.show()
