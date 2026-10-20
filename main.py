"""Generate data, train Heston parameter models, and compare calibration methods."""

import argparse
import time

import matplotlib
import numpy as np
import QuantLib as ql
import xgboost as xgb
from scipy.optimize import brentq, differential_evolution
from scipy.stats import norm, qmc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from config import (
    CALIBRATION_ERROR_PENALTY,
    CALIBRATION_MAX_ITERATIONS,
    CALIBRATION_SAMPLE_COUNT,
    DATA_CHECKPOINT_INTERVAL,
    DATA_DIR,
    DEFAULT_SAMPLE_COUNT,
    DIVIDEND_YIELD,
    FEATURE_IMPORTANCE_PLOT_PATH,
    LOWER_BOUNDS,
    MATURITIES,
    MAX_IMPLIED_VOLATILITY,
    MODEL_RANDOM_SEED,
    MONEYNESS,
    PARAMETER_BOUNDS,
    PARAMETER_NAMES,
    RANDOM_SEED,
    RISK_FREE_RATE,
    ROBUSTNESS_MAX_ITERATIONS,
    SCATTER_PLOT_PATH,
    SPOT_PRICE,
    STRIKES,
    TRAIN_TEST_FRACTION,
    UPPER_BOUNDS,
    VALIDATION_FRACTION_OF_TRAINING,
    VALUATION_DATE,
    XGBOOST_COLUMN_SUBSAMPLE,
    XGBOOST_EARLY_STOPPING_ROUNDS,
    XGBOOST_ESTIMATORS,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_SUBSAMPLE,
)


def black_scholes_call_price(spot, strike, maturity, rate, volatility):
    """Return a European call price under the Black-Scholes model."""
    d1 = (np.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * np.sqrt(maturity)
    )
    d2 = d1 - volatility * np.sqrt(maturity)
    return spot * norm.cdf(d1) - strike * np.exp(-rate * maturity) * norm.cdf(d2)


def implied_volatility(price, spot, strike, maturity, rate):
    """Find the Black-Scholes volatility corresponding to an option price."""
    try:
        return brentq(
            lambda volatility: black_scholes_call_price(
                spot, strike, maturity, rate, volatility
            )
            - price,
            0.01,
            5.0,
        )
    except (ValueError, RuntimeError, OverflowError):
        return np.nan


def heston_price(spot, strike, maturity, rate, kappa, theta, sigma, rho, v0):
    """Price a European call option with QuantLib's analytic Heston engine."""
    today = ql.Date(VALUATION_DATE.day, VALUATION_DATE.month, VALUATION_DATE.year)
    ql.Settings.instance().evaluationDate = today
    maturity_date = today + ql.Period(round(maturity * 365), ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    option = ql.VanillaOption(payoff, ql.EuropeanExercise(maturity_date))
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, rate, ql.Actual365Fixed())
    )
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, DIVIDEND_YIELD, ql.Actual365Fixed())
    )
    process = ql.HestonProcess(
        rate_handle, dividend_handle, spot_handle, v0, kappa, theta, sigma, rho
    )
    model = ql.HestonModel(process)
    option.setPricingEngine(ql.AnalyticHestonEngine(model))
    return option.NPV()


def _dataset_paths(sample_count):
    return DATA_DIR / f"X_{sample_count}.npy", DATA_DIR / f"y_{sample_count}.npy"


def create_training_data(sample_count=DEFAULT_SAMPLE_COUNT, overwrite=False):
    """Load a saved dataset or generate and save implied-volatility surfaces."""
    if sample_count <= 0:
        raise ValueError("sample_count must be a positive integer")

    features_path, targets_path = _dataset_paths(sample_count)
    if not overwrite and features_path.exists() and targets_path.exists():
        features = np.load(features_path)
        targets = np.load(targets_path)
        feature_count = len(STRIKES) * len(MATURITIES)
        if (
            features.ndim == 2
            and targets.ndim == 2
            and features.shape[1] == feature_count
            and targets.shape[1] == len(PARAMETER_NAMES)
            and features.shape[0] == targets.shape[0]
        ):
            print(f"Loaded saved dataset: X{features.shape}, y{targets.shape}")
            return features, targets
        print("Saved dataset has unexpected dimensions; regenerating it.")

    print(f"Generating {sample_count} synthetic samples...")
    sampler = qmc.LatinHypercube(d=len(PARAMETER_NAMES), seed=RANDOM_SEED)
    parameter_samples = qmc.scale(
        sampler.random(n=sample_count), LOWER_BOUNDS, UPPER_BOUNDS
    )
    all_surfaces = []
    all_parameters = []
    failed = 0
    started_at = time.perf_counter()

    for sample_index, parameters in enumerate(parameter_samples, start=1):
        surface = []
        valid = True
        for maturity in MATURITIES:
            for strike in STRIKES:
                try:
                    price = heston_price(
                        SPOT_PRICE, strike, maturity, RISK_FREE_RATE, *parameters
                    )
                    volatility = implied_volatility(
                        price, SPOT_PRICE, strike, maturity, RISK_FREE_RATE
                    )
                    if (
                        np.isnan(volatility)
                        or volatility <= 0
                        or volatility > MAX_IMPLIED_VOLATILITY
                    ):
                        valid = False
                        break
                    surface.append(volatility)
                except Exception as error:
                    print(f"Sample {sample_index} failed: {error}")
                    valid = False
                    break
            if not valid:
                break

        if valid and len(surface) == len(STRIKES) * len(MATURITIES):
            all_surfaces.append(surface)
            all_parameters.append(parameters)
        else:
            failed += 1

        if sample_index % DATA_CHECKPOINT_INTERVAL == 0:
            elapsed = time.perf_counter() - started_at
            print(
                f"Sample {sample_index}/{sample_count} | {len(all_surfaces)} good | "
                f"{failed} failed | {elapsed:.0f}s"
            )
            np.save(DATA_DIR / "X_partial.npy", np.asarray(all_surfaces))
            np.save(DATA_DIR / "y_partial.npy", np.asarray(all_parameters))

    features = np.asarray(all_surfaces, dtype=float).reshape(
        -1, len(STRIKES) * len(MATURITIES)
    )
    targets = np.asarray(all_parameters, dtype=float).reshape(-1, len(PARAMETER_NAMES))
    np.save(features_path, features)
    np.save(targets_path, targets)
    elapsed = time.perf_counter() - started_at
    print(
        f"Finished: {len(features)} successful, {failed} failed, {elapsed:.0f}s. "
        f"Saved X{features.shape} and y{targets.shape}."
    )
    return features, targets


def train_models(features, targets):
    """Split data, fit the five XGBoost models, and calculate test metrics."""
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must both be two-dimensional")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets must have the same sample count")
    if targets.shape[1] != len(PARAMETER_NAMES):
        raise ValueError(f"targets must contain {len(PARAMETER_NAMES)} columns")

    x_train_full, x_test, y_train_full, y_test = train_test_split(
        features, targets, test_size=TRAIN_TEST_FRACTION, random_state=MODEL_RANDOM_SEED
    )
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_full,
        y_train_full,
        test_size=VALIDATION_FRACTION_OF_TRAINING,
        random_state=MODEL_RANDOM_SEED,
    )
    models = {}
    metrics = {}
    for index, name in enumerate(PARAMETER_NAMES):
        print(f"Training model for {name}...")
        model = xgb.XGBRegressor(
            n_estimators=XGBOOST_ESTIMATORS,
            max_depth=XGBOOST_MAX_DEPTH,
            learning_rate=XGBOOST_LEARNING_RATE,
            subsample=XGBOOST_SUBSAMPLE,
            colsample_bytree=XGBOOST_COLUMN_SUBSAMPLE,
            early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS,
            random_state=MODEL_RANDOM_SEED,
        )
        model.fit(
            x_train,
            y_train[:, index],
            eval_set=[(x_validation, y_validation[:, index])],
            verbose=False,
        )
        predictions = model.predict(x_test)
        actual = y_test[:, index]
        models[name] = model
        metrics[name] = {
            "MAE": mean_absolute_error(actual, predictions),
            "MSE": mean_squared_error(actual, predictions),
            "R2": r2_score(actual, predictions),
        }

    print(
        f"Training samples: {len(x_train)} | Validation samples: {len(x_validation)} | "
        f"Test samples: {len(x_test)}"
    )
    return {
        "models": models,
        "metrics": metrics,
        "X_test": x_test,
        "y_test": y_test,
        "X_train": x_train,
        "y_train": y_train,
    }


def calibration_error(parameters, target_volatilities):
    """Return squared error between model and target volatility surfaces."""
    model_volatilities = []
    for maturity in MATURITIES:
        for strike in STRIKES:
            try:
                price = heston_price(
                    SPOT_PRICE, strike, maturity, RISK_FREE_RATE, *parameters
                )
                volatility = implied_volatility(
                    price, SPOT_PRICE, strike, maturity, RISK_FREE_RATE
                )
                if np.isnan(volatility):
                    return CALIBRATION_ERROR_PENALTY
                model_volatilities.append(volatility)
            except Exception:
                return CALIBRATION_ERROR_PENALTY
    return float(np.sum((np.asarray(model_volatilities) - target_volatilities) ** 2))


def predict_parameters(models, feature_row):
    """Predict parameters and clamp them to their configured ranges."""
    feature_row = np.asarray(feature_row).reshape(1, -1)
    values = [float(models[name].predict(feature_row)[0]) for name in PARAMETER_NAMES]
    return np.clip(values, LOWER_BOUNDS, UPPER_BOUNDS)


def surface_rmse(parameters, target_volatilities):
    """Calculate RMSE between target and repriced implied-volatility surfaces."""
    model_volatilities = []
    for maturity in MATURITIES:
        for strike in STRIKES:
            try:
                price = heston_price(
                    SPOT_PRICE, strike, maturity, RISK_FREE_RATE, *parameters
                )
                model_volatilities.append(
                    implied_volatility(price, SPOT_PRICE, strike, maturity, RISK_FREE_RATE)
                )
            except Exception:
                model_volatilities.append(np.nan)
    return float(
        np.sqrt(np.nanmean((np.asarray(model_volatilities) - target_volatilities) ** 2))
    )


def _traditional_calibration(target_volatilities, max_iterations, tolerance=None):
    options = {
        "func": calibration_error,
        "bounds": PARAMETER_BOUNDS,
        "args": (target_volatilities,),
        "maxiter": max_iterations,
        "seed": RANDOM_SEED,
    }
    if tolerance is not None:
        options["tol"] = tolerance
    return differential_evolution(**options)


def _run_benchmark(models, features_test, sample_count, max_iterations, tolerance=None):
    sample_count = min(sample_count, len(features_test))
    results = {
        "xgboost_times": [],
        "traditional_times": [],
        "xgboost_errors": [],
        "traditional_errors": [],
    }
    for index in range(sample_count):
        target_volatilities = features_test[index]
        started_at = time.perf_counter()
        parameters = predict_parameters(models, target_volatilities)
        results["xgboost_times"].append(time.perf_counter() - started_at)
        results["xgboost_errors"].append(surface_rmse(parameters, target_volatilities))

        started_at = time.perf_counter()
        optimum = _traditional_calibration(
            target_volatilities, max_iterations, tolerance
        )
        results["traditional_times"].append(time.perf_counter() - started_at)
        results["traditional_errors"].append(surface_rmse(optimum.x, target_volatilities))
        print(
            f"Sample {index + 1}/{sample_count}: "
            f"XGBoost RMSE={results['xgboost_errors'][-1]:.4f} | "
            f"Traditional RMSE={results['traditional_errors'][-1]:.4f}"
        )
    return results


def run_calibration_benchmark(
    models, features_test, sample_count=CALIBRATION_SAMPLE_COUNT
):
    """Run the standard XGBoost versus differential-evolution benchmark."""
    print("Comparing XGBoost with traditional calibration")
    return _run_benchmark(
        models, features_test, sample_count, CALIBRATION_MAX_ITERATIONS
    )


def run_robustness_benchmark(
    features_test, models, sample_count=CALIBRATION_SAMPLE_COUNT
):
    """Repeat the benchmark with the larger robustness iteration limit."""
    print("Running calibration robustness benchmark")
    return _run_benchmark(
        models,
        features_test,
        sample_count,
        ROBUSTNESS_MAX_ITERATIONS,
        tolerance=1e-8,
    )


def save_scatter_plot(models, metrics, features_test, targets_test):
    """Save predicted-versus-actual plots for every Heston parameter."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for index, name in enumerate(PARAMETER_NAMES):
        axis = axes[index]
        actual = targets_test[:, index]
        predicted = models[name].predict(features_test)
        axis.scatter(actual, predicted, alpha=0.3, s=10, color="steelblue")
        lower = min(actual.min(), predicted.min())
        upper = max(actual.max(), predicted.max())
        axis.plot([lower, upper], [lower, upper], "r--", linewidth=1)
        axis.set_xlabel(f"True {name}")
        axis.set_ylabel(f"Predicted {name}")
        axis.set_title(f"{name} (R2 = {metrics[name]['R2']:.3f})")
    axes[5].set_visible(False)
    figure.tight_layout()
    figure.savefig(SCATTER_PLOT_PATH, dpi=150)
    plt.close(figure)
    print(f"Saved {SCATTER_PLOT_PATH.name}")


def save_feature_importance_plot(models):
    """Save the ten most important volatility-surface features per model."""
    figure, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    feature_labels = [
        f"K={moneyness:.2f}, T={maturity:.2f}"
        for maturity in MATURITIES
        for moneyness in MONEYNESS
    ]
    for index, name in enumerate(PARAMETER_NAMES):
        axis = axes[index]
        importance = models[name].feature_importances_
        top_indices = np.argsort(importance)[-10:]
        axis.barh(range(len(top_indices)), importance[top_indices], color="steelblue")
        axis.set_yticks(range(len(top_indices)))
        axis.set_yticklabels([feature_labels[item] for item in top_indices], fontsize=8)
        axis.set_title(name)
        axis.set_xlabel("Importance")
    axes[5].set_visible(False)
    figure.suptitle("Top 10 Most Important Features per Parameter")
    figure.tight_layout()
    figure.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=150)
    plt.close(figure)
    print(f"Saved {FEATURE_IMPORTANCE_PLOT_PATH.name}")


def print_model_results(metrics):
    """Print test-set regression metrics."""
    print("\nXGBoost parameter prediction results")
    print(f"{'Parameter':<12} {'MAE':<12} {'MSE':<14} {'R2':<10}")
    for name in PARAMETER_NAMES:
        scores = metrics[name]
        print(
            f"{name:<12} {scores['MAE']:<12.4f} "
            f"{scores['MSE']:<14.6f} {scores['R2']:<10.4f}"
        )


def print_benchmark_results(results, title):
    """Print average errors and timings for one calibration benchmark."""
    if not results["xgboost_times"]:
        return
    ml_time = float(np.mean(results["xgboost_times"]))
    traditional_time = float(np.mean(results["traditional_times"]))
    speed_up = traditional_time / ml_time if ml_time else float("inf")
    print(f"\n{title}")
    print(f"Average XGBoost time:     {ml_time * 1000:.2f} ms")
    print(f"Average traditional time: {traditional_time:.2f} s")
    print(f"Speed-up factor:          {speed_up:.0f}x")
    print(f"Average XGBoost RMSE:     {np.nanmean(results['xgboost_errors']):.4f}")
    print(f"Average traditional RMSE: {np.nanmean(results['traditional_errors']):.4f}")


def run_pipeline(
    sample_count=DEFAULT_SAMPLE_COUNT,
    run_calibration=True,
    run_robustness=True,
):
    """Execute the analysis workflow and save its diagnostic figures."""
    features, targets = create_training_data(sample_count)
    if len(features) < 10:
        raise ValueError("At least 10 valid samples are required to train and evaluate")
    training = train_models(features, targets)
    print_model_results(training["metrics"])
    save_scatter_plot(
        training["models"], training["metrics"], training["X_test"], training["y_test"]
    )
    save_feature_importance_plot(training["models"])

    benchmark = None
    if run_calibration:
        benchmark = run_calibration_benchmark(training["models"], training["X_test"])
        print_benchmark_results(benchmark, "Calibration comparison")
    if run_robustness:
        robustness = run_robustness_benchmark(
            training["X_test"], training["models"]
        )
        print_benchmark_results(robustness, "Robustness comparison")
    return {"training": training, "calibration": benchmark}


def parse_arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLE_COUNT,
        help=f"number of synthetic samples to load or generate (default: {DEFAULT_SAMPLE_COUNT})",
    )
    parser.add_argument(
        "--skip-calibration",
        action="store_true",
        help="skip differential-evolution calibration",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="skip the longer calibration robustness benchmark",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    run_pipeline(
        sample_count=arguments.samples,
        run_calibration=not arguments.skip_calibration,
        run_robustness=not arguments.skip_robustness,
    )