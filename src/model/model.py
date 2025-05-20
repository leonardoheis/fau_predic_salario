from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.dummy import DummyRegressor
import numpy as np
import optuna


def train_et_model(X, y, n_trials: int = 25):
    """
    Train an ExtraTreesRegressor with Optuna hyperparameter tuning,
    evaluating with RMSE, MAE, MAPE and R² via 5-fold CV.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Feature matrix.
    y : array-like, shape (n_samples,)
        Target vector.
    n_trials : int, default=25
        Number of Optuna trials.

    Returns
    -------
    best_model : ExtraTreesRegressor
        Re-fitted on the full dataset with the best parameters.
    metrics : dict
        Cross-validated metrics on the best model.
        Keys are 'RMSE','MAE','MAPE','R2'.
    """
    def objective(trial):
        # define search space
        params = {
            'n_estimators':       trial.suggest_int('n_estimators', 100, 1000),
            'max_depth':          trial.suggest_int('max_depth', 3, 20),
            'min_samples_split':  trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf':   trial.suggest_int('min_samples_leaf', 1, 20),
            'max_features':       trial.suggest_categorical('max_features',
                                                           ['sqrt', 'log2', None])
        }
        model = ExtraTreesRegressor(
            random_state=42,
            n_jobs=-1,
            **params
        )
        # evaluate multiple metrics
        scoring = {
            'rmse': 'neg_root_mean_squared_error',
            'mae' : 'neg_mean_absolute_error',
            'mape': 'neg_mean_absolute_percentage_error',
            'r2'  : 'r2'
        }
        cv_results = cross_validate(
            model, X, y,
            cv=5,
            scoring=scoring,
            return_train_score=False
        )
        # we minimize RMSE, so return its positive mean
        return -cv_results['test_rmse'].mean()

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # retrain on full data
    best_model = ExtraTreesRegressor(
        random_state=42,
        n_jobs=-1,
        **best_params
    ).fit(X, y)

    # final cross-val metrics
    scoring = {
        'rmse': 'neg_root_mean_squared_error',
        'mae' : 'neg_mean_absolute_error',
        'mape': 'neg_mean_absolute_percentage_error',
        'r2'  : 'r2'
    }
    cv_final = cross_validate(
        best_model, X, y,
        cv=5,
        scoring=scoring,
        return_train_score=False
    )
    metrics = {
        'RMSE': -cv_final['test_rmse'].mean(),
        'MAE' : -cv_final['test_mae'].mean(),
        'MAPE': -cv_final['test_mape'].mean(),
        'R2'  : cv_final['test_r2'].mean()
    }
    print("Cross-val metrics:", metrics)

    return best_model, metrics


def evaluate_model(model, X_test, y_test, show_examples=True, baseline=True):
    """
    Evaluates a trained model using MAE and RMSE (with confidence intervals).
    Also compares it against a DummyRegressor baseline.

    Args:
        model (object): Trained regression model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True target values (log-salaries).
        show_examples (bool): Whether to print sample predictions (currently disabled).
        baseline (bool): Whether to compare against a DummyRegressor.

    Returns:
        None
    """

    #y_pred_log = model.predict(X_test)
    #y_pred_real = np.exp(y_pred_log)
    #y_test_real = np.exp(y_test)
    y_pred_real = model.predict(X_test)
    y_test_real = y_test
    
    mae  = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    mape = mean_absolute_percentage_error(y_test_real, y_pred_real)
    r2   = r2_score(y_test_real, y_pred_real)
    
    # bootstrap CIs
    mae_ci  = bootstrap_metrics(y_test_real.values, y_pred_real, mean_absolute_error)
    rmse_ci = bootstrap_metrics(y_test_real.values, y_pred_real,
                       lambda y, yhat: np.sqrt(mean_squared_error(y, yhat)))
    mape_ci = bootstrap_metrics(y_test_real.values, y_pred_real, mean_absolute_percentage_error)
    r2_ci   = bootstrap_metrics(y_test_real.values, y_pred_real, r2_score)

    print("\n performance on test:")
    print("----------------------------------")
    print(" Model Evaluation Summary:")
    print("----------------------------------")
    print(f"MAE: ${mae:,.2f} (on average, predictions deviate this much from actual salaries)")
    print(f"95% Confidence Interval for MAE: ${mae_ci[0]:,.2f} – ${mae_ci[1]:,.2f}\n")

    print(f"RMSE: ${rmse:,.2f} ")
    print(f"95% Confidence Interval for RMSE: ${rmse_ci[0]:,.2f} – ${rmse_ci[1]:,.2f}\n")
    
    print(f"MAPE: {mape:.2%} ")
    print(f"95% Confidence Interval for MAPE: {mape_ci[0]:.2%} – {mape_ci[1]:.2%}\n")
    
    print(f"R²  : {r2:.3f} ")
    print(f"95% Confidence Interval for R²: {r2_ci[0]:.3f} – {r2_ci[1]:.3f}\n")

    '''
    #ONLY for testing purposes
    if show_examples:
        print("\n Comparation of real salaries (test vs prediction):")
        for real, pred in zip(y_test_real[:5], y_pred_real[:5]):
            print(f"Real: ${real:,.2f}  |  Predicted: ${pred:,.2f}")
    '''
    if baseline:
        dummy = DummyRegressor(strategy="mean")
        dummy.fit(X_test, y_test)
        dummy_preds = dummy.predict(X_test)

        dummy_mae = mean_absolute_error(y_test_real, dummy_preds)
        dummy_rmse = np.sqrt(mean_squared_error(y_test_real, dummy_preds))
        dummy_mape = mean_absolute_percentage_error(y_test_real, dummy_preds)
        dummy_r2   = r2_score(y_test_real, dummy_preds)

    improvement_mae = dummy_mae - mae
    improvement_rmse = dummy_rmse - rmse
    improvement_mape = dummy_mape - mape
    improvement_r2   = r2 - dummy_r2
    print("Comparison vs baseline (dum. regressor using mean):")
    print("--------------------------------------------------------")
    print(f" MAE (Dummy): ${dummy_mae:,.2f} -  Improvement: ${improvement_mae:,.2f}")
    print(f" RMSE (Dummy): ${dummy_rmse:,.2f}  -  Improvement: ${improvement_rmse:,.2f}")
    print(f" MAPE (Dummy): {dummy_mape:.2%}  —  Improvement: {improvement_mape:.2%}")
    print(f" R² (Dummy): {dummy_r2:.3f}  —  Improvement: {improvement_r2:.3f}")
    
    return y_pred_real

def bootstrap_metrics(y_true, y_pred, metric_fn, n_bootstrap=1000, ci=95):
    """
    Computes a confidence interval for a metric using bootstrap resampling.

    Args:
        y_true (np.array): True target values.
        y_pred (np.array): Predicted values.
        metric_fn (function): Metric function to apply (e.g., mean_absolute_error).
        n_bootstrap (int): Number of bootstrap samples.
        ci (int): Confidence interval percentage (e.g., 95).

    Returns:
        tuple: Lower and upper bounds of the confidence interval.
    """

    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(range(n), size=n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    alpha = (100 - ci) / 2
    lower = np.percentile(scores, alpha)
    upper = np.percentile(scores, 100 - alpha)

    return lower, upper