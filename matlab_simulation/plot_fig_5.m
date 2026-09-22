script_dir = fileparts(mfilename('fullpath'));
reachability_dir = fullfile(script_dir, '..', 'cora_reachability', 'Reachability');

tiledlayout("flow"); nexttile

Alsomitra_Control_Simulation(fullfile(reachability_dir,"adversarial_model_005_denorm.onnx"),"PID Controller",false)

nexttile

Alsomitra_Control_Simulation(fullfile(reachability_dir,"base_model_denorm.onnx"),"NN Controller",true)

legend("Simulated Trajectory","Desired Trajectory")
