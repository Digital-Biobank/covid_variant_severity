# %% Imports
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %% Read in outcomes data
df = pd.read_csv("data/2020-10-21_green-red.csv", index_col=0)

# Figure S1
df["covv_patient_status"] = df["covv_patient_status"].str.lower().str.strip()
red_status = df.reset_index().groupby(
    ["is_red", "covv_patient_status"]
)["pid"].count().reset_index().sort_values(
    ["is_red", "pid"], ascending=False
)
red_status = red_status[red_status["pid"] > 19]
colors = pd.Series(["red" if r else "blue" for r in red_status["is_red"]])
bar = sns.barplot(x="pid", y="covv_patient_status", data=red_status, palette=colors)
# Define some hatches
hatches = colors.str.replace("red", "x").str.replace("blue", "+")
# Loop over the bars
for i,thisbar in enumerate(bar.patches):
    # Set a different hatch for each bar
    thisbar.set_hatch(hatches[i])
plt.xlabel("Patient Count")
plt.ylabel("Status")
plt.title("Patient count by status")
plt.tight_layout()
plt.savefig(f"plots/2020-10-21_fig-s1.png", dpi=300)
plt.show()
