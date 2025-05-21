import seaborn as sns
import matplotlib.pyplot as plt

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
