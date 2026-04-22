# Environment and Set Up

I use UV for environment management and package installation in this project. If you have UV installed, you can get up and running with the existing environment with:

```bash
uv sync
```

Alternatively, you can manually install the required packages listed in `pyproject.toml` to your active python environment using pip:

```bash
pip install .
```

# Running the Training Code

Before training, make sure you've download and extracted the FashionMNIST dataset.

```bash
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz 
```

The training script accepts a config file. An example config is provided in `configs/example_config.yaml`. Modify the parameters to match your data directory and desired experiment settings. Then run the training loop by executing the `dfa_net.py` script with the path to your config file for the `--config` argument:

```bash
# For example, with UV:
uv run dfa_net.py --config "configs/example_config.yaml"
```

# Visualizations

Loss curve visualizations may be generated in the `visualizations.ipynb` notebook. Change the `experiment_output_dir` variable to match the path to our experiment output directory.