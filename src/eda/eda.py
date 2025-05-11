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

def check_nulls(df, name="DataFrame"):
    """
    Prints the number of null values per column in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame.
        name (str): Label to identify the DataFrame in output.
    """


    print(f"\n amount of nulls per column {name}:")
    print(df.isnull().sum())
    
def count_salary_nulls(df):
    """
    Counts and prints how many rows have null values in the Salary column.

    Args:
        df (pd.DataFrame): DataFrame with a Salary column.
    """

    null_salary = df["Salary"].isnull().sum()
    print(f"\n rows in salary that are null: {null_salary}")

def count_rows_with_any_null(df, name="DataFrame"):
    """
    Prints how many rows have at least one null value and shows them.

    Args:
        df (pd.DataFrame): Input DataFrame.
        name (str): Label to identify the DataFrame in output.
    """

    null_rows = df[df.isnull().any(axis=1)]
    print(f"\n amount of rows that have at least a NaN value: {len(null_rows)}")
    print(null_rows)

def plot_distributions(df):
    """
    Plots histograms for Salary, Age, Years of Experience, and log(Salary).

    Args:
        df (pd.DataFrame): DataFrame that includes those columns.
    """

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))

    sns.histplot(df["Salary"], bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Salary")

    sns.histplot(df["Age"], bins=20, kde=True, ax=axes[1])
    axes[1].set_title("Age")

    sns.histplot(df["Years of Experience"], bins=20, kde=True, ax=axes[2])
    axes[2].set_title("Years of Experience")

    sns.histplot(df["Salary_log"], bins=30, kde=True, ax=axes[3])
    axes[3].set_title("ln(Salary)")

    plt.tight_layout()
    plt.show()