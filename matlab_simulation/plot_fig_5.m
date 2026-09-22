tiledlayout("flow"); nexttile

Alsomitra_Control_Simulation("adversarial_model_005_denorm.onnx","PID Controller",false)

nexttile

Alsomitra_Control_Simulation("base_model_denorm.onnx","NN Controller",true)

legend("Simulated Trajectory","Desired Trajectory")