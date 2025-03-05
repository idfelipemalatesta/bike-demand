import math
import pandas as pd
import xgboost as xgb
import mlflow
import optuna
from sklearn.metrics import mean_squared_error
from utils.eda_utils import plot_feature_importance, plot_residuals
from utils.modeling_utils import champion_callback
import warnings

warnings.filterwarnings("ignore")


def load_data(
    train_path="data/processed/train.csv", valid_path="data/processed/valid.csv"
):
    train = pd.read_csv(train_path)
    valid = pd.read_csv(valid_path)

    return train, valid


def preprocess(train, valid):
    train_x = train.drop(columns="cnt")
    train_y = train["cnt"]
    valid_x = valid.drop(columns="cnt")
    valid_y = valid["cnt"]

    dtrain = xgb.DMatrix(train_x, label=train_y)
    dvalid = xgb.DMatrix(valid_x, label=valid_y)

    return train_x, train_y, valid_x, valid_y, dtrain, dvalid


def get_experiment_id(experiment_name="Bike Demand"):
    # If experiment exists, return its ID; otherwise, create one.
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def objective(trial, dtrain, valid_y, dvalid):
    with mlflow.start_run(nested=True):
        # Define hyperparameters
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "booster": trial.suggest_categorical(
                "booster", ["gbtree", "gblinear", "dart"]
            ),
            "lambda": trial.suggest_float("lambda", 1e-8, 1.0, log=True),
            "alpha": trial.suggest_float("alpha", 1e-8, 1.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.2, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
        }

        if params["booster"] in ["gbtree", "dart"]:
            params["max_depth"] = trial.suggest_int("max_depth", 1, 9)
            params["min_child_weight"] = trial.suggest_int("min_child_weight", 2, 10)
            params["learning_rate"] = trial.suggest_float(
                "learning_rate", 1e-8, 1.0, log=True
            )
            params["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)

        bst = xgb.train(params, dtrain, num_boost_round=200)
        preds = bst.predict(dvalid)
        error = mean_squared_error(valid_y, preds)

        mlflow.log_params(params)
        mlflow.log_metric("mse", error)
        mlflow.log_metric("rmse", math.sqrt(error))
        return error


def run_optimization(dtrain, dvalid, valid_y, n_trials=100):
    study = optuna.create_study(direction="minimize")

    def objective_func(trial):
        return objective(trial, dtrain, valid_y, dvalid)

    study.optimize(objective_func, n_trials=n_trials, callbacks=[champion_callback])
    return study


def log_best_model(study, dtrain, dvalid, valid_y):
    mlflow.log_params(study.best_params)
    mlflow.log_metric("best_mse", study.best_value)
    mlflow.log_metric("best_rmse", math.sqrt(study.best_value))

    mlflow.set_tags(
        {
            "project": "Bike Demand Project",
            "optimizer_engine": "optuna",
            "model_family": "xgboost",
            "feature_set_version": 1,
        }
    )

    # Train the final model using the best parameters
    model = xgb.train(study.best_params, dtrain)

    # Log feature importances plot
    fig_importance = plot_feature_importance(
        model, booster=study.best_params.get("booster")
    )
    mlflow.log_figure(figure=fig_importance, artifact_file="feature_importances.png")

    # Log residuals plot
    fig_residuals = plot_residuals(model, dvalid, valid_y)
    mlflow.log_figure(figure=fig_residuals, artifact_file="residuals.png")

    artifact_path = "model"
    mlflow.xgboost.log_model(
        xgb_model=model,
        artifact_path=artifact_path,
        model_format="ubj",
        metadata={"model_data_version": 1},
    )

    model_uri = mlflow.get_artifact_uri(artifact_path)
    return model_uri


def main():
    # Optionally, adjust verbosity for other libraries if needed
    optuna.logging.set_verbosity(optuna.logging.ERROR)

    # Load the data
    train, valid = load_data()

    # Prepare features, target and create DMatrix objects
    train_x, train_y, valid_x, valid_y, dtrain, dvalid = preprocess(train, valid)

    # Configure MLflow tracking and experiment
    mlflow.set_tracking_uri("http://localhost:5000")
    experiment_id = get_experiment_id("Bike Demand v4")
    mlflow.set_experiment(experiment_id=experiment_id)
    run_name = "100trails_200num_boost"

    # Start the parent run
    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name, nested=True):
        # Run hyperparameter optimization with Optuna
        study = run_optimization(dtrain, dvalid, valid_y, n_trials=100)

        # Log the best model and associated artifacts
        model_uri = log_best_model(study, dtrain, dvalid, valid_y)
        print("Model:", model_uri)


if __name__ == "__main__":
    main()
