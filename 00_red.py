# %% Imports
import json
import pandas as pd

recode_dict = {
    "Hospitalized": 1,
    # "Live": 0,
    # "Live ": 0,
    # "live": 0,
    # "Alive": 0,
    # "Symptomatic": 1,
    # "symptomatic": 1,
    "Deceased": 1,
    "Outpatient": 0,
    "Asymptomatic": 0,
    "Mild": 0,
    "Mild clinical signs without hospitalization": 0,
    "Home": 0,
    "Hospitalized (Severe)": 1,
    "Hospitalized (Moderate)": 1,
    "Hospitalized; Stable": 1,
    "Not hospitalized": 0,
    "hospitalized or to be hospitalized": 1,
    "hospitalized": 1,
    "Not Hospitalized": 0,
    "Quarantine": 0,
    "Ward": 1,
    "Hospitalized, Live": 1,
    "Intensive Care Unit": 1,
    "Hospitalized (Critical)": 1,
    "Still hospitalized": 1,
    "Death": 1,
    "Severe / ICU": 1,
    "Contact with and exposure to other communicable diseases": 0,
    "No clinical signs": 0,
    "No clinical signs without hospitalization": 0,
    "Moderate / Outpatient": 0,
    "ICU": 1,
    "Mild / Contact exposure / Asymptomatic": 0,
    "ICU; Serious": 1,
    "Hospitalized/Released": 1,
    "Hospitalized, deceased": 1,
    "Quarantined": 0,
    "outpatient": 0,
    "Benigne": 0,
    "inpatient": 1,
    "Hospitalized, released": 1,
    "Encounter for general adult medical examination": 0,
    "Hospitalised": 1,
    "Hospitalized/Deceased": 1,
    "Facility quarantine": 0,
    "Inpatient": 1,
    " Deceased": 1,
    "Hospitaized": 1,
    "Mild case": 0,
    "Stable in quarantine": 0,
    "Hospitalized": 1,
    "Deceased": 1,
    "Asympomatic": 0,
    "Mild, at home.": 0,
    "Epidemiology Study": 0,
    "Hospitalized, oxygenotherapy, diarrhea": 1,
    "Hospitalized (Intensive care unit)": 1,
    "Isolation": 0,
    "Screening": 0,
    "Hospsitalized, ICU, fully recovered": 1,
    "Pneumonia, unspecified organism": 1,
    "Hospitalized or to be hospitalized": 1,
    "Asymptomatic/Released": 0,
    "Encounter for observation for other suspected diseases and conditions ruled out": 0,
    "Live, mild symptoms, at home": 0,
    "Mild symptoms (fever, cardiovascular disorders)": 0,
    " Hospitalized": 1,
    "Hospitalization": 1,
    "Asymptomatic, identified as positive during preoperation investigation": 0,
    "deceased": 1,
    "asymptomatic": 0,
}

def select_dicts(filename):
    with open(filename, "r") as f:
        for l in f.readlines():
            d = json.loads(l)
            if d.get("covv_patient_status") in recode_dict.keys():
                yield d

df = pd.DataFrame(select_dicts("data/2020-10-21.json"))

df = df.assign(
    is_red=df["covv_patient_status"].map(recode_dict),
    pid=df["covv_accession_id"].str.extract(
        r"(\d+)",
        expand=False
    ).astype(int)
).set_index("pid")

cols = ["continent", "country", "region", "city"]

df[cols] = df["covv_location"].str.split("/", expand=True).iloc[:, :-1]

df.to_parquet("data/2020-10-21_green-red.parquet")
df.to_csv("data/2020-10-21_green-red.csv")