# SAIV_2025_Alsomitra
Codebase for [Neural Network Verification for Gliding Drone Control: A Case Study](https://link.springer.com/chapter/10.1007/978-3-031-99991-8_9)

This folder contains scripts for training and verifying a neural network control system for a small bio-inspired gliding drone, actuated by changing the position of the centre of mass (CoM), with behaviour cloning. The diaspore in question is _Alsomitra macrocarpa_, modelled with a quasi-steady 2D aerodynamic model for falling plates with displaced CoM.

# Overview
**System requirements**
- MATLAB r2024a - for system simulation and reachability verification (CORA v2025.1.0)
- Python 3.9 - for network training and onnx manipulation
- Vehicle/Marabou - for network verification

**Repository layout**
- `python_training/` - train regression networks, export ONNX models, and prepare verification data
- `vehicle_verification/` - Vehicle specifications, verification models, IDX data, and outputs
- `matlab_simulation/` - MATLAB simulations used to generate the training data
- `cora_reachability/` - CORA reachability experiments
- `docs/` - paper and slides
- `Figures/` - figure assets for documentation and the paper

# Part 1 - Generating Data in MATLAB

The first step in the full workflow is to run a set of drone simulations in MATLAB, in order to generate training data. This code works with MATLAB r2024a - **if the onnx converter causes issues I suggest using MATLAB online**. Start by cloning this repository, and adding **all folders and subfolders** to the MATLAB path.

The control simulations can be run with the following command, where the inputs are the controller NN (not used in this case), plot title, and NN controller switch (set to `false`, since we are using a PID controller).

```matlab
Alsomitra_Control_Simulation(fullfile("python_training","models","Baseline.onnx"),"PID Controller",false)
```

This generates a plot of simulation traces which should follow the target trajectory, and a dataset in csv and MAT formats under `matlab_simulation/data/` (`Training_Data.csv` and `Training_Data.mat`).

The training simulations run for 20 seconds with a control frequency of 0.5 s, and since we record the system states and controller output for each control action, this gives 40 datapoints per trajectory - for a total of 360 datapoints. Each data point consists of 6 system states which serve as NN inputs, and a controller output which our NN will try to predict.

However, for effective adversarial training, this data needs to be normalised between 0 and 1. This is achieved with a min max normalisation script (`matlab_simulation/Normalise_Data.m`), generating an equivalent normalised dataset (`Training_Data_Normalised.csv` and `Training_Data_Normalised.mat`) in `python_training/data/`. Dealing with normalised and system values is a core limitation of this work.

# Part 2 - NN Training in Python

Create a Python 3.9 environment, then install the required packages with:

```bash
pip install -r requirements.txt
```

The training script is `python_training/main.py`, and it should run with no issues, generating NNs in `.onnx` format under `python_training/models/`. Specifically, the script is designed to train a naive "baseline" model, and a robust "adversarial" model (with a certain epsilon value) to be compared.

Under the hood, the baseline model is trained as a regression network on the normalised dataset, with the goal of predicting the controller output from the 6 system states. A second model is then trained with adversarial examples generated in an epsilon-ball around the training data. This is the key robust-training step used in the paper, and is intended to improve local robustness of the controller while preserving useful regression performance.

There is an extra step required. Since the training data is normalised (required for adversarial training), the networks produced in the most recent step deal with normalised values, so will not work properly in simulation. I solve this by adding an extra layer to the input and output of the network when it is in ONNX format, to **normalise the input and denormalise the output**. The script is called `python_training/normalise_network.py`, and requires 2 things to be implemented:

- The NN to be used (`model_path`) and output name.
- The normalisation constants `Cs` and `Ss`. These are generated when normalising the data with `Normalise_Data.m`, so can be copied from the MATLAB workspace.

`python_training/Export_IDX.py` converts the normalised training data into IDX format for Vehicle-based verification.

# Part 3 - Verification in Vehicle

The relevant files are:
- `vehicle_verification/property_verification/` - global properties and associated models/data
- `vehicle_verification/property_verification_2/` - additional verification experiments
- `vehicle_verification/data/` - shared IDX datasets
- `vehicle_verification/results/` - verification outputs

The Vehicle specifications cover:
- global properties relating the controller output to the desired trajectory
- local robustness properties over epsilon-balls around training data

# Part 4 - Running Inference / Simulations in MATLAB

Once trained, the network can be imported back to MATLAB to test regression accuracy and control performance. The NN controller functions as a drop-in replacement for the PID code in the initial simulations, so simulations are run similarly to before:

```matlab
Alsomitra_Control_Simulation(fullfile("cora_reachability","Reachability","base_model_denorm.onnx"),"NN Controller",true)
```

Another script compares the regression performance over the training dataset for multiple networks: `matlab_simulation/Check_NN_Accuracy.m`.

# Part 5 - Reachability with CORA (MATLAB)

The reachability workflow is kept in `cora_reachability/` as part of the full case-study workflow described in the paper.

# Further Background

The repository also includes:
- `docs/paper/` - manuscript sources and PDF
- `docs/slides/` - presentation slides

If you want the broader context for the case study, start with the paper in `docs/paper/`.
