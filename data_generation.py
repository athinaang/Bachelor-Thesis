"""Parameter sampling and synthetic dataset generation from the notebook."""

import time

import numpy as np
from scipy.stats import qmc

from config import (
    DATA_CHECKPOINT_INTERVAL,
    DATA_DIR,
    DEFAULT_SAMPLE_COUNT,
    LOWER_BOUNDS,
    MATURITIES,
    MAX_IMPLIED_VOLATILITY,
    PARAMETER_NAMES,
    RISK_FREE_RATE,
    SPOT_PRICE,
    STRIKES,
    UPPER_BOUNDS,
)
from pricing import heston_price, implied_vol


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

            # Each sample contains 7 maturities x 10 strikes in surface_for order.
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
                # Preserve progress in case the long generation run is interrupted.
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