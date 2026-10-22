"""Run the Heston thesis workflow."""

from analysis import (
    plot_feature_importance,
    plot_predictions,
    print_final_summary,
    print_results_table,
    run_calibration_benchmark,
    run_robustness_benchmark,
    run_xgboost_rmse_check,
    split_dataset,
    train_models,
)
from data_generation import generate_or_load_dataset, generate_parameter_samples
from pricing import test_heston_pricer


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