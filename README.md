# SAIV_2025_Alsomitra
Codebase for Neural Network Verification for Gliding Drone Control: A Case Study

(25/7/25) - I am aware this repo is terribly disorganised, and will work on tidying it up asap - if you would like help running this code send me an email - ck2049@hw.ac.uk. In the longer term I plan to format it into a VNN-COMP benchmark for future competitions

# Overview
**System requirements**
- MATLAB r2024a - for system simulation, and reachability compuation (CORA v2025.1.0)
- Python 3.9 - for NN training, and onnx manipulation
- Vehicle/Marabou - for NN verification

**MATLAB scripts:**
- ODEs (nondimfreelyfallingplate)
- Control Simulation (Alsomitra_Control_Simulation)
- NN inference (Check_NN_Accuracy)
- data normalisation (Normalise_Data)

**Python scripts:**
- NN training (main, adversarial_trainer, model_builder, progress_bar, evaluator)
- onnx merging (merge_onnx_models)
- onnx denormalisation (normalise_network)
- data export for verfication (Export_IDX)

**Vehicle scripts**
- Global properties (Alsomitra)
- Local Robustness (Alsomitra_Robustness)

# Part 1 - Running simulations in MATLAB

# Part 2 - Training NN in Python

# Part 3 - Running inference / simulations in MATLAB

# Part 4 - Verification in Vehicle

# Part 5 - Reachability with CORA (MATLAB)
