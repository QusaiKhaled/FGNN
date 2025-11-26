import pandas as pd

# Read the original CSV file
df = pd.read_csv("2018_SCADA_Pressures.csv", sep=";", engine="python")

# Convert timestamp format
df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# Convert column n1 to float by replacing comma with dot
df["n1"] = df["n1"].str.replace(",", ".", regex=False).astype(float)

# Select the last three months (starting from October 1, 2018)
start = pd.Timestamp("2018-10-01")
mask = df["Timestamp"] >= start

# Create a copy of the dataframe for modification
df_out = df.copy()

# Add incremental drift to n1 during the last three months:
# 0.0001, 0.0002, 0.0003, ... applied sequentially
indices = df_out[mask].index
for k, i in enumerate(indices):
    df_out.at[i, "n1"] = df_out.at[i, "n1"] + (k + 1) * 0.0001

# Save the modified dataframe to a new CSV file
df_out.to_csv("2018_SCADA_Pressure_Drift.csv", sep=";", index=False)
