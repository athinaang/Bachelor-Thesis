# Heston Model Calibration Thesis Code

This project generates Heston implied-volatility surfaces, trains one XGBoost model for each Heston parameter, and compares machine-learning predictions with differential-evolution calibration. The original [thesis_code.ipynb](thesis_code.ipynb) notebook is kept as a research reference; its functions are grouped by purpose across the Python modules.

## Requirements

- Python 3.12 or 3.13
- Git (only needed to upload the project)

## Setup

Open a terminal in the project directory and create a virtual environment.

### Windows PowerShell

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If Python 3.13 is not installed, use `py -3.12 -m venv .venv` instead. If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again.

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

Run the complete workflow, including both calibration benchmarks:

```bash
python main.py
```

On Windows, `python .\main.py` is equivalent. The program loads `X_20000.npy` if that file exists; otherwise it generates 20,000 samples and saves `X_20000.npy` and `y_20000.npy`. The full workflow includes numerical optimization and can take a long time. Delete both saved dataset files before running if you want to regenerate them. Generated datasets and plots are excluded from Git by `.gitignore`. The program saves `scatter_plots.png` and `feature_importance.png` in the project directory.

## Project files

- `main.py`: short entry point that runs the workflow in notebook order.
- `pricing.py`: Heston pricing, Black-Scholes pricing, and implied volatility.
- `data_generation.py`: parameter sampling and synthetic dataset generation/loading.
- `analysis.py`: model training, calibration benchmarks, plots, and summary tables.
- `config.py`: shared fixed values used by the modules.
- `thesis_code.ipynb`: preserved notebook version of the research workflow.

