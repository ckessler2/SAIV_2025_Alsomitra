[completed1, R1,simRes1, dims1] = Alsomitra_Closed_Loop_Reachability("base_model_denorm.onnx",0,16);
% save 'base_0.mat' R1
% 
% [completed2, R2,simRes2, dims2] = Alsomitra_Closed_Loop_Reachability("adversarial_model_005_denorm.onnx",0,8);
% save '005_0.mat' R2