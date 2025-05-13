import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import shap

# Set global seaborn style
sns.set_style("darkgrid")


def boxplots(data, numerical_cols):
    """
    Creates boxplots for the specified numerical columns with mean lines.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the variables to visualize.
    numerical_cols : list of str
        List of numerical column names to plot.
    """
    num_vars = len(numerical_cols)
    cols = 3
    rows = math.ceil(num_vars / cols)

    plt.figure(figsize=(6 * cols, 5 * rows))
    plt.suptitle('Distribution of Each Variable with Mean', fontsize=16, y=1.02)

    for i, col in enumerate(numerical_cols):
        plt.subplot(rows, cols, i + 1)
        sns.boxplot(data=data, x=col, width=0.5)
        plt.axvline(x=data[col].mean(), color='r', linestyle='--', label='Mean')
        plt.title(col)
        plt.legend()

    plt.tight_layout()
    plt.show()


def create_plots(columns, plot_func, n_cols=3):
    """
    Creates a grid of subplots and applies a plotting function to each column.

    Parameters
    ----------
    columns : list of str
        List of column names to plot.
    plot_func : function
        Function that takes a column name and matplotlib axis and generates a plot.
    n_cols : int, optional
        Number of columns per row in the subplot grid. Default is 3.
    """
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))

    axes = np.ravel(axes) if isinstance(axes, np.ndarray) else [axes]

    for i, col in enumerate(columns):
        plot_func(col, axes[i])
        axes[i].set_title(f'Distribution of {col}', fontsize=12)
        axes[i].set_xlabel(col, fontsize=10)

    for j in range(len(columns), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    plt.show()


def scatterplot(df: pd.DataFrame, target: str = 'Salary', exclude: list = None):
    """
    Creates a grid of scatterplots showing the relationship between each feature and the target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the features and target.
    target : str
        Name of the target column.
    exclude : list of str, optional
        List of columns to exclude from the plots.
    """
    if exclude is None:
        exclude = ['id', 'Description', 'Job Title', target]

    cols = df.columns.difference(exclude)
    n_cols = 2
    n_rows = math.ceil(len(cols) / n_cols)

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))
    for i, col in enumerate(cols):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.scatterplot(data=df, x=col, y=target)
        plt.title(f'{col} vs {target}')
        plt.xlabel(col)
        plt.ylabel(target)
        plt.tight_layout()

    plt.suptitle(f'Relationship Between Features and Target ({target})', fontsize=16, y=1.02)
    plt.show()


def plot_shap_feature_importance(model, X_test_final, X_test, embedding_prefix='embedding_', plot_type='bar'):
    """
    Computes and plots SHAP feature importances, grouping embedding features.

    Parameters
    ----------
    model : trained model
        A fitted tree-based model compatible with SHAP.
    X_test_final : np.ndarray
        Test set used for prediction (after transformation, includes embeddings).
    X_test : pd.DataFrame
        Original test set before embeddings were added.
    embedding_prefix : str
        Prefix for identifying embedding features. Default is 'embedding_'.
    plot_type : str
        Type of SHAP summary plot ('bar' or 'dot'). Default is 'bar'.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_final)

    original_features = X_test.columns.tolist()
    num_original = len(original_features)

    if len(np.array(shap_values).shape) == 3:  # Multi-class
        shap_values_agg = np.concatenate([
            shap_values[:, :, :num_original],
            np.sum(shap_values[:, :, num_original:], axis=2, keepdims=True)
        ], axis=2)
    else:  # Single output
        shap_values_agg = np.concatenate([
            shap_values[:, :num_original],
            np.sum(shap_values[:, num_original:], axis=1, keepdims=True)
        ], axis=1)

    all_features = original_features + ['Text_Embeddings']

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_agg, X_test_final[:, :num_original + 1],
                      feature_names=all_features, plot_type=plot_type)

def plot_shap_feature_importance_no_embeddings(model, X_test_final, X_test, plot_type='bar'):
    """
    Computes and plots SHAP feature importances, EXCLUDING embedding features.

    Parameters
    ----------
    model : trained model
        A fitted tree-based model compatible with SHAP.
    X_test_final : np.ndarray
        Test set used for prediction (post-transformation, may include embeddings).
    X_test : pd.DataFrame
        Original test set (pre-embedding transformation).
    plot_type : str
        Type of SHAP summary plot ('bar' or 'dot'). Default is 'bar'.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_final)

    original_features = X_test.columns.tolist()
    num_original = len(original_features)

    # Extract SHAP values only for original features (exclude embeddings)
    if len(np.array(shap_values).shape) == 3:  # Multi-class
        shap_values_agg = shap_values[:, :, :num_original]
    else:  # Single output
        shap_values_agg = shap_values[:, :num_original]

    # Plot SHAP summary (only non-embedding features)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_agg, 
        X_test_final[:, :num_original],  # Original feature data
        feature_names=original_features, 
        plot_type=plot_type,
        show=False
    )
    plt.title("SHAP Feature Importance (Excluding Embeddings)", fontsize=14)
    plt.tight_layout()
    plt.show()