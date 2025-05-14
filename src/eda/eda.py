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

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.histplot(df["Salary"], bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Salary")

    sns.histplot(df["Age"], bins=20, kde=True, ax=axes[1])
    axes[1].set_title("Age")

    sns.histplot(df["Years of Experience"], bins=20, kde=True, ax=axes[2])
    axes[2].set_title("Years of Experience")

    # sns.histplot(df["Salary_log"], bins=30, kde=True, ax=axes[3])
    # axes[3].set_title("ln(Salary)")

    plt.tight_layout()
    plt.show()
    
def missing_data_table(data):
    """
    Generate a table showing the count and percentage of missing (NaN) values in each column.

    Parameters
    data : pd.DataFrame
        The DataFrame to analyze for missing values.

    Returns
    pd.DataFrame
        A DataFrame containing the number and percentage of NaN values per column.
    """
    count = data.isna().sum()
    percentage = round(count / len(data) * 100, 2)
    result = {
        'NaN Count': count,
        'NaN Percentage (%)': percentage
    }
    table = pd.DataFrame(result)
    return table

def percentage_rows_missing_data(df):
    """
    Calculates the percentage of rows with at least one missing value.

    Args:
        df: pandas DataFrame

    Returns:
        float: Percentage of rows with missing data.
    """

    rows_with_missing_data = df.isnull().any(axis=1).sum()
    total_rows = len(df)
    percentage = (rows_with_missing_data / total_rows) * 100
    return percentage

def print_df_overview(df, name="DataFrame"):
    """
    Prints basic information about a DataFrame: number of rows/columns and data types.

    Args:
        df (pd.DataFrame): Input DataFrame.
        name (str): Label to identify the DataFrame in output.
    """

    print("\n dtype:")
    print(df.dtypes)
    
def count_job_titles(df, threshold):
    """
    Displays the count of job titles that appear more than a given threshold.

    Args:
        df (pd.DataFrame): DataFrame with a 'Job Title' column.
        threshold (int): Minimum number of appearances to be considered.
    """

    job_counts = df["Job Title"].value_counts()
    titles_above_N = job_counts[job_counts > threshold]
    total_rows_above_N = titles_above_N.sum()

    print(f"\n Amount of rows with job titles that appear more than {threshold} times: {total_rows_above_N}")
    print(f" Job Titles with more than {threshold} repetitions:\n{titles_above_N}")
    
def plot_salary_by_gender(df):
    """
    Plots a boxplot to visualize salary distribution across genders.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame with 'Gender' and 'Salary'.
    """
    plt.figure(figsize=(8,5))
    sns.boxplot(x="Gender", y="Salary", data=df)
    plt.title("Salary Distribution by Gender")
    plt.ylabel("Salary")
    plt.xlabel("Gender")
    plt.show()

def gender_salary_stats(df):
    """
    Prints mean salary and count per gender.
    
    Args:
        df (pd.DataFrame): Cleaned DataFrame with 'Gender' and 'Salary'.
    """
    grouped = df.groupby("Gender")["Salary"].agg(["count", "mean", "median"])
    print("Salary stats by gender:")
    print(grouped)
    

def plot_kde_salary_by_gender(df):
    """
    Plots KDE curves to compare salary distributions between genders.
    """
    plt.figure(figsize=(8,5))
    sns.kdeplot(data=df, x="Salary", hue="Gender", fill=True)
    plt.title("Salary distribution by Gender")
    plt.xlabel("Salary")
    plt.ylabel("Count")
    plt.show()
    print("Note: If the KDE appears to go below zero, it is a plotting artifact. Actual density values are not negative.")


