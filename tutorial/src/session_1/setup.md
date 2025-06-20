# Setting up the Environment

Before you can run code for this tutorial, you need to set up your environment.

This section walks you through:
1. Getting the tutorial code.
2. Setting up your Python environment.
3. Verifying everything is working.

## Step 1: Clone the Tutorial Repository

Clone the code repository from GitHub:

```bash
git clone https://github.com/ALPHA-g-Experiment/ml-tutorial.git
cd ml-tutorial/code
```

## Step 2: Set Up Your Python Environment

The easiest way to set up your environment is to
[install uv](https://docs.astral.sh/uv/getting-started/installation/#installation-methods).

After installing uv, you can create and activate a virtual environment with the
following commands:

```bash
uv venv --python 3.12
source .venv/bin/activate
```

Finally, install the required packages:

```bash
uv pip install -r requirements.txt
```

## Step 3: Verify the Setup

To verify that everything is set up correctly, run the following command:

```bash
python -c "import torch; print(torch.__version__)"
```

If you see the version of PyTorch printed without any errors, your setup is
successful.
