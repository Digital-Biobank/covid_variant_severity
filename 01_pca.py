# %% Imports
import joblib
import pandas as pd
from sklearn.decomposition import PCA

# %% Read in VCF wide data
df = pd.read_parquet("00_77142-vcf_wide.parquet").fillna(0)

# %% Instantiate PCA model
pca = PCA(n_components=2, svd_solver="randomized", random_state=42)

# %% Fit PCA model and create transformed data for plotting
transformed = pca.fit_transform(df)

# %% Add patient IDs (pid) index to transformed data
transformed.index = df.index

# %% Label transformed data columns as PC1 and PC2
transformed.columns = ["PC1", "PC2"]

# %% Save transformed data for plotting
transformed.to_csv("01_77142-vcf_2-component-pca-transformed.csv")
transformed.to_parquet("01_77142-vcf_2-component-pca-transformed.parquet")

# %% Save PCA model
joblib.dump(pca, "01_77142-vcf_2-component-pca-model.pickle.gz")

# %% Put PCA components (loadings) into a dataframe
pca_df = pd.DataFrame({
    "PC1": pca.components_[0],
    "PC2": pca.components_[1]
    },
    index=df.columns
)

# %% Save PCA components (loadings)
pca_df.to_csv("01_77142-vcf_2-component-pca-components.csv")
pca_df.to_parquet("01_77142-vcf_2-component-pca-components.parquet")

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

# %% Filter out other variants
pca_plot_data = pca_df.loc[variants]

# %% Save top variant PCA component correlations
pca_plot_data.T.to_csv(
    "01_77142-vcf_2-component-pca-components_top-variants_wide.csv"
)
pca_plot_data.to_csv(
    "01_77142-vcf_2-component-pca-components_top-variants_long.csv"
)
