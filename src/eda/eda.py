import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def merge_multiple_dataframes(file_paths, on_column):
    """
    Merges multiple dataframes provided via file paths on a specified common column.

    Parameters:
    file_paths (list of str): List containing paths to the CSV files.
    on_column (str): Column name to merge on.

    Returns:
    pd.DataFrame: Merged dataframe.
    """
    if not file_paths:
        raise ValueError("No file paths provided")

    # Load the first dataframe
    merged_df = pd.read_csv(file_paths[0])

    # Iteratively merge remaining dataframes
    for path in file_paths[1:]:
        next_df = pd.read_csv(path)
        merged_df = merged_df.merge(next_df, on=on_column)

    return merged_df