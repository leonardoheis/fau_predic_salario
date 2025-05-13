import sqlite3
import os
import seaborn as sns
import matplotlib.pyplot as plt

# def save_predictions_to_database(df, db_path="predictions.db", table_name="predictions"):
#     '''
#     Save predictions to a SQLite database.
#     If the database does not exist, it will be created.
#     args:
#         df (pd.DataFrame): DataFrame containing the predictions to save.
#         db_path (str): Path to the SQLite database file.
#         table_name (str): Name of the table to save the predictions in.  
#     '''
#     new_database = not os.path.exists(db_path)
#     connection = sqlite3.connect(db_path)

#     df.to_sql(table_name, connection, if_exists="append", index=False)

#     connection.close()

#     print(f"Saved rows to '{db_path}' in table '{table_name}'")
    
def plot_numeric_feature_vs_target(df, feature, target="Salary"):
    """
    Plots a scatterplot between a numeric feature and the target variable.
    The target variable is assumed to be the log of the salary.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        feature (str): Name of the numeric feature to plot against the target.
        target (str): Name of the target variable (default is "Salary_log").
        
    """
    sns.scatterplot(data=df, x=feature, y=target)
    plt.title(f"{target} vs {feature}")
    plt.tight_layout()
    plt.show()


def plot_categorical_feature_vs_target(df, feature, target="Salary"):
    """
    Plots a boxplot between a categorical feature and the target variable.
    The target variable is assumed to be the log of the salary.
    
    args:
        df (pd.DataFrame): DataFrame containing the data to plot.
        feature (str): Name of the categorical feature to plot against the target.
        target (str): Name of the target variable (default is "Salary_log").
    """
    sns.boxplot(data=df, x=feature, y=target)
    plt.xticks(rotation=45)
    plt.title(f"{target} by {feature}")
    plt.tight_layout()
    plt.show()
