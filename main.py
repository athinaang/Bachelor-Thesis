"""Run the Heston pricing and parameter calibration thesis workflow."""

import time

import matplotlib.pyplot as plt
import numpy as np
import QuantLib as ql
import xgboost as xgb
from scipy.optimize import brentq, differential_evolution
from scipy.stats import norm, qmc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from config import (
    CALIBRATION_ITERATIONS,
    CALIBRATION_SAMPLE_COUNT,
    DATA_CHECKPOINT_INTERVAL,
    DATA_DIR,
    DEFAULT_SAMPLE_COUNT,
    DIVIDEND_YIELD,
    FEATURE_IMPORTANCE_PLOT_PATH,
    LOWER_BOUNDS,
    MATURITIES,
    MAX_IMPLIED_VOLATILITY,
    MONEYNESS,
    PARAMETER_NAMES,
    PARAMETER_RANGES,
    RANDOM_STATE,
    RISK_FREE_RATE,
    ROBUSTNESS_ITERATIONS,
    SCATTER_PLOT_PATH,
    SPOT_PRICE,
    STRIKES,
    TRAIN_TEST_FRACTION,
    VALUATION_DATE,
    UPPER_BOUNDS,
    VALIDATION_FRACTION_OF_TRAINING,
    XGBOOST_COLUMN_SUBSAMPLE,
    XGBOOST_EARLY_STOPPING_ROUNDS,
    XGBOOST_ESTIMATORS,
    XGBOOST_LEARNING_RATE,
    XGBOOST_MAX_DEPTH,
    XGBOOST_SUBSAMPLE,
)


def heston_price(S, K, T, r, kappa, theta, sigma, rho, v0):
    """Price a European call option under the Heston model."""
    today = ql.Date(VALUATION_DATE.day, VALUATION_DATE.month, VALUATION_DATE.year)
    ql.Settings.instance().evaluationDate = today
    maturity_date = today + ql.Period(round(T * 365), ql.Days)

    payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
    exercise = ql.EuropeanExercise(maturity_date)
    option = ql.VanillaOption(payoff, exercise)

    spot_handle = ql.QuoteHandle(ql.SimpleQuote(S))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, r, ql.Actual365Fixed())
    )
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, DIVIDEND_YIELD, ql.Actual365Fixed())
    )

    heston_process = ql.HestonProcess(
        rate_handle, dividend_handle, spot_handle, v0, kappa, theta, sigma, rho
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    option.setPricingEngine(engine)
    return option.NPV()


def test_heston_pricer():
    """Run the pricing examples and Black-Scholes verification from the notebook."""
    price = heston_price(
        S=SPOT_PRICE,
        K=SPOT_PRICE,
        T=1.0,
        r=RISK_FREE_RATE,
        kappa=2.0,
        theta=0.04,
        sigma=0.5,
        rho=-0.7,
        v0=0.04,
    )
    print(f"Heston call price: {price:.4f}")
    print(
        heston_price(
            S=SPOT_PRICE,
            K=SPOT_PRICE,
            T=5,
            r=RISK_FREE_RATE,
            kappa=2.0,
            theta=0.04,
            sigma=0.5,
            rho=-0.7,
            v0=0.04,
        )
    )

    S, K, T, r, vol = 100, 100, 1.0, 0.03, 0.20
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    bs_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    heston_test = heston_price(
        S=100,
        K=100,
        T=1.0,
        r=0.03,
        kappa=2.0,
        theta=0.04,
        sigma=0.001,
        rho=0.0,
        v0=0.04,
    )
    print(f"Black-Scholes price: {bs_price:.4f}")
    print(f"Heston price:        {heston_test:.4f}")
    print(f"Difference:          {abs(bs_price - heston_test):.6f}")


def bs_call_price(S, K, T, r, sigma):
    """Black-Scholes call price given volatility sigma."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def implied_vol(price, S, K, T, r):
    """Find Black-Scholes implied volatility for a given option price."""
    try:
        iv = brentq(
            lambda sigma: bs_call_price(S, K, T, r, sigma) - price,
            0.01,
            5.0,
        )
        return iv
    except:
        return np.nan


def generate_parameter_samples():
    """Generate the notebook's Latin-hypercube parameter samples."""
    sampler = qmc.LatinHypercube(d=5)
    samples_unit = sampler.random(n=DEFAULT_SAMPLE_COUNT)
    samples_scaled = qmc.scale(samples_unit, LOWER_BOUNDS, UPPER_BOUNDS)

    print("First 3 parameter sets:")
    print(f"{'kappa':>8} {'theta':>8} {'sigma':>8} {'rho':>8} {'v0':>8}")
    for i in range(3):
        kappa, theta, sigma, rho, v0 = samples_scaled[i]
        print(f"{kappa:8.3f} {theta:8.4f} {sigma:8.3f} {rho:8.3f} {v0:8.4f}")
    return samples_scaled


def surface_for(kappa, theta, sigma, rho, v0):
    """Calculate a complete implied-volatility surface for Heston parameters."""
    today = ql.Date(VALUATION_DATE.day, VALUATION_DATE.month, VALUATION_DATE.year)
    ql.Settings.instance().evaluationDate = today
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(SPOT_PRICE))
    rate_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, RISK_FREE_RATE, ql.Actual365Fixed())
    )
    dividend_handle = ql.YieldTermStructureHandle(
        ql.FlatForward(today, DIVIDEND_YIELD, ql.Actual365Fixed())
    )
    option_type = ql.Option.Call

    options_list = []
    for T in MATURITIES:
        for K in STRIKES:
            maturity_date = today + ql.Period(round(T * 365), ql.Days)
            payoff = ql.PlainVanillaPayoff(option_type, K)
            exercise = ql.EuropeanExercise(maturity_date)
            option = ql.VanillaOption(payoff, exercise)
            options_list.append((option, K, T))

    heston_process = ql.HestonProcess(
        rate_handle, dividend_handle, spot_handle, v0, kappa, theta, sigma, rho
    )
    heston_model = ql.HestonModel(heston_process)
    engine = ql.AnalyticHestonEngine(heston_model)
    results = []
    for option, K, T in options_list:
        option.setPricingEngine(engine)
        price = option.NPV()
        iv = implied_vol(price, SPOT_PRICE, K, T, RISK_FREE_RATE)
        results.append(iv)
    return results


def generate_or_load_dataset(samples_scaled):
    """Load the notebook dataset when present or generate it otherwise."""
    features_path = DATA_DIR / f"X_{DEFAULT_SAMPLE_COUNT}.npy"
    targets_path = DATA_DIR / f"y_{DEFAULT_SAMPLE_COUNT}.npy"

    if features_path.exists():
        X = np.load(features_path)
        y = np.load(targets_path)
        print(f"Loaded from disk: X{X.shape}, y{y.shape}")
    else:
        print(f"No saved data found. Generating {DEFAULT_SAMPLE_COUNT} samples...")
        all_ivs = []
        all_params = []
        failed = 0
        start_time = time.time()

        for i in range(DEFAULT_SAMPLE_COUNT):
            kappa, theta, sigma, rho, v0 = samples_scaled[i]
            iv_surface = []
            success = True

            for T in MATURITIES:
                for K in STRIKES:
                    try:
                        price = heston_price(
                            SPOT_PRICE, K, T, RISK_FREE_RATE,
                            kappa, theta, sigma, rho, v0
                        )
                        iv = implied_vol(price, SPOT_PRICE, K, T, RISK_FREE_RATE)
                        if np.isnan(iv) or iv <= 0 or iv > MAX_IMPLIED_VOLATILITY:
                            print(
                                "rejected: iv =", iv, "| K =", K, "| T =", T,
                                "| v0 =", v0, "| rho =", rho
                            )
                            success = False
                            break
                        iv_surface.append(iv)
                    except Exception as e:
                        print(e)
                        success = False
                        break

                if not success:
                    break

            if success and len(iv_surface) == 70:
                all_ivs.append(iv_surface)
                all_params.append([kappa, theta, sigma, rho, v0])
            else:
                failed += 1

            if (i + 1) % DATA_CHECKPOINT_INTERVAL == 0:
                elapsed = time.time() - start_time
                print(
                    f"Sample {i+1}/{DEFAULT_SAMPLE_COUNT} | {len(all_ivs)} good | "
                    f"{failed} failed | {elapsed:.0f}s"
                )
                np.save(DATA_DIR / "X_partial.npy", np.array(all_ivs))
                np.save(DATA_DIR / "y_partial.npy", np.array(all_params))

        elapsed = time.time() - start_time
        print(
            f"\nFinished: {len(all_ivs)} successful, {failed} failed, {elapsed:.0f}s"
        )
        X = np.array(all_ivs)
        y = np.array(all_params)
        np.save(features_path, X)
        np.save(targets_path, y)
        print(f"Saved: X{X.shape}, y{y.shape}")
    return X, y


def split_dataset(X, y):
    """Create the notebook's train, validation, and test splits."""
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=TRAIN_TEST_FRACTION, random_state=RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_FRACTION_OF_TRAINING,
        random_state=RANDOM_STATE,
    )
    print(f"Training set:   {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set:       {X_test.shape[0]} samples")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_models(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train the notebook's five XGBoost models and calculate metrics."""
    models = {}
    results = {}
    for i, name in enumerate(PARAMETER_NAMES):
        print(f"\nTraining model for: {name}")
        model = xgb.XGBRegressor(
            n_estimators=XGBOOST_ESTIMATORS,
            max_depth=XGBOOST_MAX_DEPTH,
            learning_rate=XGBOOST_LEARNING_RATE,
            subsample=XGBOOST_SUBSAMPLE,
            colsample_bytree=XGBOOST_COLUMN_SUBSAMPLE,
            early_stopping_rounds=XGBOOST_EARLY_STOPPING_ROUNDS,
            random_state=RANDOM_STATE,
        )
        model.fit(
            X_train,
            y_train[:, i],
            eval_set=[(X_val, y_val[:, i])],
            verbose=False,
        )
        y_pred = model.predict(X_test)
        y_true = y_test[:, i]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        models[name] = model
        results[name] = {"MAE": mae, "MSE": mse, "R2": r2}
        print(f"  MAE = {mae:.4f}")
        print(f"  MSE = {mse:.6f}")
        print(f"  R2  = {r2:.4f}")
    print("\n--- DONE ---")
    return models, results


def print_results_table(results):
    """Print the notebook's regression results table."""
    print("=" * 55)
    print(f"{'Parameter':<12} {'MAE':<12} {'MSE':<14} {'R²':<10}")
    print("=" * 55)
    for name in PARAMETER_NAMES:
        metrics = results[name]
        print(
            f"{name:<12} {metrics['MAE']:<12.4f} "
            f"{metrics['MSE']:<14.6f} {metrics['R2']:<10.4f}"
        )
    print("=" * 55)


def plot_predictions(models, results, X_test, y_test):
    """Create and save the notebook's predicted-versus-true scatter plots."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for i, name in enumerate(PARAMETER_NAMES):
        ax = axes[i]
        y_true = y_test[:, i]
        y_pred = models[name].predict(X_test)
        ax.scatter(y_true, y_pred, alpha=0.3, s=10, color="steelblue")
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f'{name}  (R² = {results[name]["R2"]:.3f})')
    axes[5].set_visible(False)
    plt.tight_layout()
    plt.savefig(SCATTER_PLOT_PATH, dpi=150)
    plt.show()
    print("Saved: scatter_plots.png")


def calibration_error(params, target_ivs, S, r, strikes, maturities):
    """Calculate squared error for traditional Heston calibration."""
    kappa, theta, sigma, rho, v0 = params
    model_ivs = []
    for T in maturities:
        for K in strikes:
            try:
                price = heston_price(S, K, T, r, kappa, theta, sigma, rho, v0)
                iv = implied_vol(price, S, K, T, r)
                if np.isnan(iv):
                    return 1e6
                model_ivs.append(iv)
            except Exception as e:
                print(e)
                return 1e6
    error = np.sum((np.array(model_ivs) - np.array(target_ivs)) ** 2)
    return error


def run_calibration_benchmark(models, X_test, y_test):
    """Run the notebook's five-sample calibration comparison."""
    n_test = CALIBRATION_SAMPLE_COUNT
    bounds = list(zip(LOWER_BOUNDS, UPPER_BOUNDS))
    print("Comparing XGBoost vs Traditional Calibration")
    print("=" * 70)
    xgb_times = []
    trad_times = []
    xgb_errors = []
    trad_errors = []

    for j in range(n_test):
        target_ivs = X_test[j]
        true_params = y_test[j]

        t0 = time.time()
        xgb_params = [
            float(models[name].predict(X_test[j:j + 1])[0])
            for name in PARAMETER_NAMES
        ]
        xgb_time = time.time() - t0
        xgb_times.append(xgb_time)

        xgb_ivs = []
        for T in MATURITIES:
            for K in STRIKES:
                try:
                    p = heston_price(SPOT_PRICE, K, T, RISK_FREE_RATE, *xgb_params)
                    iv = implied_vol(p, SPOT_PRICE, K, T, RISK_FREE_RATE)
                    xgb_ivs.append(iv if not np.isnan(iv) else 0)
                except Exception as e:
                    print(e)
                    xgb_ivs.append(0)
        xgb_err = np.sqrt(np.mean((np.array(xgb_ivs) - target_ivs) ** 2))
        xgb_errors.append(xgb_err)

        t0 = time.time()
        result = differential_evolution(
            calibration_error,
            bounds=bounds,
            args=(target_ivs, SPOT_PRICE, RISK_FREE_RATE, STRIKES, MATURITIES),
            maxiter=CALIBRATION_ITERATIONS,
            seed=RANDOM_STATE,
        )
        trad_time = time.time() - t0
        trad_times.append(trad_time)

        trad_params = result.x
        trad_ivs = []
        for T in MATURITIES:
            for K in STRIKES:
                try:
                    p = heston_price(SPOT_PRICE, K, T, RISK_FREE_RATE, *trad_params)
                    iv = implied_vol(p, SPOT_PRICE, K, T, RISK_FREE_RATE)
                    trad_ivs.append(iv if not np.isnan(iv) else 0)
                except Exception as e:
                    print(e)
                    trad_ivs.append(0)
        trad_err = np.sqrt(np.mean((np.array(trad_ivs) - target_ivs) ** 2))
        trad_errors.append(trad_err)
        print(
            f"Sample {j+1}: XGBoost {xgb_time*1000:.1f}ms (RMSE={xgb_err:.4f}) | "
            f"Traditional {trad_time:.1f}s (RMSE={trad_err:.4f})"
        )

    print("=" * 70)
    print(f"\nAverage XGBoost time:     {np.mean(xgb_times)*1000:.1f} ms")
    print(f"Average Traditional time: {np.mean(trad_times):.1f} s")
    print(f"Speed-up factor:          {np.mean(trad_times)/np.mean(xgb_times):.0f}x faster")
    print(f"\nAverage XGBoost RMSE:     {np.mean(xgb_errors):.4f}")
    print(f"Average Traditional RMSE: {np.mean(trad_errors):.4f}")
    return xgb_times, trad_times, xgb_errors, trad_errors


def run_xgboost_rmse_check(models, X_test):
    """Run the notebook's XGBoost surface RMSE check for every test row."""
    xgb_rmses = []
    for j in range(len(X_test)):
        target_ivs = X_test[j]
        params = [
            float(models[name].predict(X_test[j:j + 1])[0])
            for name in PARAMETER_NAMES
        ]
        params = list(np.clip(params, LOWER_BOUNDS, UPPER_BOUNDS))
        ivs = surface_for(*params)
        rmse = np.sqrt(np.mean((np.array(ivs) - target_ivs) ** 2))
        xgb_rmses.append(rmse)
    print("samples:", len(xgb_rmses))
    print("excluded (nan):", np.isnan(xgb_rmses).sum())
    print("average RMSE:", np.nanmean(xgb_rmses))
    print("median RMSE: ", np.nanmedian(xgb_rmses))
    print("worst RMSE:  ", np.nanmax(xgb_rmses))
    return xgb_rmses


def run_robustness_benchmark(X_test, models):
    """Run the notebook's differential-evolution maxiter=200 check."""
    n_test = CALIBRATION_SAMPLE_COUNT
    bounds = list(zip(LOWER_BOUNDS, UPPER_BOUNDS))
    print("Precise Timing Comparison (robustness check, maxiter=200)")
    print("=" * 70)
    xgb_times_robust = []
    trad_times_robust = []
    xgb_errors_robust = []
    trad_errors_robust = []

    for j in range(n_test):
        target_ivs = X_test[j]
        t0 = time.perf_counter()
        xgb_params = [
            float(models[name].predict(X_test[j:j + 1])[0])
            for name in PARAMETER_NAMES
        ]
        xgb_time = time.perf_counter() - t0
        xgb_times_robust.append(xgb_time)

        xgb_ivs = []
        for T in MATURITIES:
            for K in STRIKES:
                try:
                    p = heston_price(SPOT_PRICE, K, T, RISK_FREE_RATE, *xgb_params)
                    iv = implied_vol(p, SPOT_PRICE, K, T, RISK_FREE_RATE)
                    xgb_ivs.append(iv if not np.isnan(iv) else 0)
                except Exception as e:
                    print(e)
                    xgb_ivs.append(0)
        xgb_err = np.sqrt(np.mean((np.array(xgb_ivs) - target_ivs) ** 2))
        xgb_errors_robust.append(xgb_err)

        t0 = time.perf_counter()
        result = differential_evolution(
            calibration_error,
            bounds=bounds,
            args=(target_ivs, SPOT_PRICE, RISK_FREE_RATE, STRIKES, MATURITIES),
            maxiter=ROBUSTNESS_ITERATIONS,
            tol=1e-8,
            seed=RANDOM_STATE,
        )
        trad_time = time.perf_counter() - t0
        trad_times_robust.append(trad_time)

        trad_ivs = []
        for T in MATURITIES:
            for K in STRIKES:
                try:
                    p = heston_price(SPOT_PRICE, K, T, RISK_FREE_RATE, *result.x)
                    iv = implied_vol(p, SPOT_PRICE, K, T, RISK_FREE_RATE)
                    trad_ivs.append(iv if not np.isnan(iv) else 0)
                except Exception as e:
                    print(e)
                    trad_ivs.append(0)
        trad_err = np.sqrt(np.mean((np.array(trad_ivs) - target_ivs) ** 2))
        trad_errors_robust.append(trad_err)
        print(
            f"Sample {j+1}: XGBoost {xgb_time*1000:.2f}ms (RMSE={xgb_err:.4f}) | "
            f"Traditional {trad_time:.2f}s (RMSE={trad_err:.4f})"
        )

    print("=" * 70)
    print(f"\nAverage XGBoost time:     {np.mean(xgb_times_robust)*1000:.2f} ms")
    print(f"Average Traditional time: {np.mean(trad_times_robust):.2f} s")
    print(f"Speed-up factor:          {np.mean(trad_times_robust)/np.mean(xgb_times_robust):.0f}x faster")
    print(f"\nAverage XGBoost RMSE:     {np.mean(xgb_errors_robust):.4f}")
    print(f"Average Traditional RMSE: {np.mean(trad_errors_robust):.4f}")
    return xgb_times_robust, trad_times_robust, xgb_errors_robust, trad_errors_robust


def plot_feature_importance(models):
    """Create and save the notebook's feature-importance plots."""
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    feature_labels = []
    for T in MATURITIES:
        for m in MONEYNESS:
            feature_labels.append(f"K={m},T={T:.2f}")

    for i, name in enumerate(PARAMETER_NAMES):
        ax = axes[i]
        importance = models[name].feature_importances_
        top_10_idx = np.argsort(importance)[-10:]
        top_10_imp = importance[top_10_idx]
        top_10_labels = [feature_labels[idx] for idx in top_10_idx]
        ax.barh(range(10), top_10_imp, color="steelblue")
        ax.set_yticks(range(10))
        ax.set_yticklabels(top_10_labels, fontsize=8)
        ax.set_title(f"{name}")
        ax.set_xlabel("Importance")

    axes[5].set_visible(False)
    plt.suptitle("Top 10 Most Important Features per Parameter", fontsize=14)
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_PLOT_PATH, dpi=150)
    plt.show()
    print("Saved: feature_importance.png")


def print_final_summary(
    results, X_train, X_test, xgb_rmses, xgb_times, trad_times, trad_errors
):
    """Print the notebook's final thesis summary tables."""
    print("=" * 65)
    print("TABLE 1: XGBoost Parameter Prediction Accuracy (Test Set)")
    print("=" * 65)
    print(f"{'Parameter':<12} {'Range':<16} {'MAE':<10} {'MSE':<12} {'R²':<8}")
    print("-" * 65)
    for name in PARAMETER_NAMES:
        metrics = results[name]
        print(
            f"{name:<12} {PARAMETER_RANGES[name]:<16} "
            f"{metrics['MAE']:<10.4f} {metrics['MSE']:<12.6f} "
            f"{metrics['R2']:<8.4f}"
        )
    print("=" * 65)
    print(f"\nTraining samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]} implied volatilities")
    print("\n")
    print("=" * 65)
    print("TABLE 2: Calibration Method Comparison")
    print("=" * 65)
    print(f"{'Method':<30} {'Avg RMSE':<15} {'Avg Time':<15}")
    print("-" * 65)
    print(
        f"{'Heston (XGBoost)':<30} {np.nanmean(xgb_rmses):<15.4f} "
        f"{np.mean(xgb_times)*1000:<10.2f} ms"
    )
    print(
        f"{'Heston (Differential Evol.)':<30} {np.mean(trad_errors):<15.4f} "
        f"{np.mean(trad_times):<10.2f} s"
    )
    print("=" * 65)
    print(
        f"\nXGBoost is {np.mean(trad_times)/np.mean(xgb_times):.0f}x faster, "
        f"with an average surface RMSE of {np.nanmean(xgb_rmses):.4f} "
        f"vs {np.mean(trad_errors):.4f} for traditional calibration."
    )


def main():
    test_heston_pricer()
    samples_scaled = generate_parameter_samples()
    X, y = generate_or_load_dataset(samples_scaled)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)
    models, results = train_models(X_train, X_val, X_test, y_train, y_val, y_test)
    print_results_table(results)
    plot_predictions(models, results, X_test, y_test)

    xgb_times, trad_times, xgb_errors, trad_errors = run_calibration_benchmark(
        models, X_test, y_test
    )
    xgb_rmses = run_xgboost_rmse_check(models, X_test)
    run_robustness_benchmark(X_test, models)
    plot_feature_importance(models)
    print_final_summary(
        results,
        X_train,
        X_test,
        xgb_rmses,
        xgb_times,
        trad_times,
        trad_errors,
    )


if __name__ == "__main__":
    main()