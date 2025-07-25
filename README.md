# SAIV_2025_Alsomitra
Codebase for Neural Network Verification for Gliding Drone Control: A Case Study

(24/7/25) - I am aware this repo is terribly disorganised, and will work on tidying it up asap. In the longer term I plan to format it into a VNN-COMP benchmark for future competitions

**System requirements**
- MATLAB - for system simulation, and reachability compuation (CORA)
- Python - for NN training, and onnx manipulation
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
