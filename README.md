# Heston Model Calibration Thesis Code

This project generates Heston implied-volatility surfaces, trains one XGBoost model for each Heston parameter, and compares machine-learning predictions with differential-evolution calibration. The original [thesis_code.ipynb](thesis_code.ipynb) notebook is kept as a research reference; the runnable workflow is organized into Python modules.

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

On Windows, `python .\main.py` is equivalent. The program loads `X_20000.npy` and `y_20000.npy` if both are present; otherwise it generates and saves the dataset. Generated datasets and plots are excluded from Git by `.gitignore`.

Options for a smaller or quicker run:

```bash
python main.py --samples 5000
python main.py --skip-calibration --skip-robustness
python main.py --skip-robustness
```

The full run includes numerical optimization and can take a long time. If the dataset is missing, generating 20,000 samples can also take a long time. To regenerate data, delete the matching `X_<sample-count>.npy` and `y_<sample-count>.npy` files first; generated data is intentionally not versioned. The program saves `scatter_plots.png` and `feature_importance.png` in the project directory.

## Project files

- `main.py`: all workflow functions and the command-line entry point.
- `config.py`: shared model, grid, data, and benchmark settings.
- `thesis_code.ipynb`: preserved notebook version of the research workflow.

## Upload to GitHub

The GitHub repository was empty when this project was prepared. The local folder is connected to it as `origin`. Run these commands from this project directory to review and upload the files:

```bash
git status
git add .
git status
git commit -m "Organize thesis code into Python modules"
git push -u origin main
```

Review the second `git status` before committing. It should include source files, README, requirements, and the notebook, but not `.venv`, generated `.npy` datasets, or generated plot files. If `git remote add origin` reports an existing remote, inspect it using `git remote -v`. Authenticate with the browser prompt from Git Credential Manager; do not put passwords or tokens in a command.