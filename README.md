# SAIV_2025_Alsomitra
Codebase for Neural Network Verification for Gliding Drone Control: A Case Study

(25/7/25) - I am aware this repo is terribly disorganised, and will work on tidying it up asap - if you would like help running this code send me an email - ck2049@hw.ac.uk. In the longer term I plan to format it into a VNN-COMP benchmark for future competitions

# Overview
**System requirements**
- MATLAB r2024a - for system simulation and reachability verification (CORA v2025.1.0)
- Python 3.9 - for network training and onnx manipulation
- Vehicle/Marabou - for network verification

**MATLAB scripts:**
- ODEs (nondimfreelyfallingplate)
- Control Simulation (Alsomitra_Control_Simulation)
- NN inference (Check_NN_Accuracy)
- data normalisation (Normalise_Data)
- Compute reachability (Alsomitra_Closed_Loop_Reachability)
- Plot reachability (plot_reachable_sets)

**Python scripts:**
- NN training (main, adversarial_trainer, model_builder, progress_bar, evaluator)
- onnx merging (merge_onnx_models)
- onnx denormalisation (normalise_network)
- data export for verfication (Export_IDX)

**Vehicle scripts**
- Global properties (Alsomitra)
- Local Robustness (Alsomitra_Robustness)

# Part 1 - Generating Data in MATLAB

The first step is to run a set of drone simulations in MATLAB, in order to generate training data. This code works with MATLAB r2024a - if the onnx converter causes issues I suggest using MATLAB online. Start by cloning this repository, and adding **all folders and subfolders** to the MATLAB path. The control simulations can be run with the following command, where the inputs are the controller NN (not used in this case), plot title, and NN controller switch (set to "false", since we are using a PID controller).

```
Alsomitra_Control_Simulation("Baseline.onnx","PID Controller","false")
```

This generates a plot of simulation traces which should follow the target trajectory, and a data file in csv and MAT formats (Training_Data.csv and Training_Data.mat).

The training simulations run for 20 seconds with a control frequency of 0.5s, and since we record the system states and controller output for each control action, this gives 40 datapoints per trajectory - for a total of 360 datapoints. Each data point (each row in the csv) consists of 6 system states which serve as NN inputs, and a controller output which our NN will try to predict.

# Part 2 - Training NN in Python

# Part 3 - Running inference / simulations in MATLAB

# Part 4 - Verification in Vehicle

# Part 5 - Reachability with CORA (MATLAB)
