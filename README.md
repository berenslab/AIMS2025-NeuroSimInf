# Simulation and Inference for Neuroscience

This repository contains the code for the Simulation and Inference for Neuroscience course taught at AIMS.

## Download

You can download the repository by running:
```bash
git clone https://github.com/aims-neuroscience/simulation-and-inference-for-neuroscience.git
```

or by clicking the "Code" button above and clicking "Download ZIP".

## Installation
There is two different ways to install the dependencies:

### Option 1: Using uv

We recommend using [uv](https://docs.astral.sh/uv/) to manage the dependencies.
If you're on macOS or Linux you can install `uv` by running:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

If you're on Windows you can install `uv` by running:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Once you have `uv` installed, you can install the dependencies, create a python virtual environment and activate it by running:

```bash
uv sync
source .venv/bin/activate
```

Then you can run the notebooks by running:

```bash
jupyter-notebook notebooks/
```

If you are using an IDE, it should be able to detect the python environment and use it automatically.
I recommend using [vscode](https://code.visualstudio.com/).

### Option 2: Using conda

Alternatively, you can use [conda](https://docs.conda.io/en/latest/) to manage the dependencies.
If you don't have conda installed, you can install it [here](https://docs.conda.io/en/latest/miniconda.html).

Once you have conda installed, you can create a python environment and install the dependencies by running:

```bash
conda create -n sim-inf-neuro python=3.10
conda activate sim-inf-neuro
pip install -e .
```

Then you can run the notebooks by running:

```bash
jupyter-notebook notebooks/
```
