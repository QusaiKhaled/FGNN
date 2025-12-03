import pandas as pd
import numpy as np


# ============================================================
# Function: Add incremental drift to selected nodes
# ============================================================
def add_incremental_drift(
    input_csv: str,
    output_csv: str,
    node_intervals: dict,
    step_range: tuple = (0.00005, 0.0002),
    sep: str = ";"
):
    """
    Apply incremental drift to specified SCADA sensor nodes.

    Parameters
    ----------
    input_csv : str
        Path to the original SCADA CSV file.
    output_csv : str
        Path where the drifted CSV will be saved.
    node_intervals : dict
        Dictionary mapping each node (column name) to a (start_time, end_time) tuple.
        Example:
            {
                "n1": ("2018-10-01", "2018-12-31"),
                "n5": ("2018-09-01", "2018-11-15")
            }
    step_range : tuple(float, float)
        Range of random drift steps (min_step, max_step).
        For each node, a random step is sampled, and drift is applied as:
            +step, +2*step, +3*step, ...
    sep : str
        CSV separator used in input/output files.
    """

    # Load original data
    df = pd.read_csv(input_csv, sep=sep, engine="python")

    # Convert timestamp column
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])

    # Make output copy
    df_out = df.copy()

    # Process each node independently
    for node, (start_str, end_str) in node_intervals.items():

        # Check if node exists
        if node not in df_out.columns:
            print(f"[Warning] Node '{node}' not found in the CSV. Skipped.")
            continue

        # Convert string values to float (support comma decimal format)
        if df_out[node].dtype == object:
            df_out[node] = (
                df_out[node].astype(str).replace(",", ".", regex=False).astype(float)
            )

        # Parse time interval
        start = pd.to_datetime(start_str)
        end = pd.to_datetime(end_str)

        # Select rows within the interval
        mask = (df_out["Timestamp"] >= start) & (df_out["Timestamp"] <= end)
        indices = df_out[mask].index

        if len(indices) == 0:
            print(f"[Info] No rows found for node {node} in [{start_str}, {end_str}].")
            continue

        # Sample a random drift step for this node
        step = np.random.uniform(step_range[0], step_range[1])
        print(f"[Info] Node {node}: applying incremental drift with step={step:.6f}, rows={len(indices)}")

        # Apply drift incrementally
        for k, i in enumerate(indices):
            df_out.at[i, node] = df_out.at[i, node] + (k + 1) * step

    # Save output CSV
    df_out.to_csv(output_csv, sep=sep, index=False)
    print(f"[Done] Drifted file saved to: {output_csv}")


# ============================================================
# Example Usage
# ============================================================

if __name__ == "__main__":

    # Define which nodes to drift and their time intervals
    drift_nodes = {
        "n1": ("2018-10-01", "2018-12-31"),
        "n3": ("2018-09-01", "2018-11-30"),
        "n7": ("2018-08-15", "2018-10-15")
    }

    # Call drift function
    add_incremental_drift(
        input_csv="2018_SCADA_Pressures.csv",
        output_csv="2018_SCADA_Pressures_Drifted.csv",
        node_intervals=drift_nodes,
        step_range=(0.00005, 0.0002),  # drift range
        sep=";"                         # CSV delimiter
    )
