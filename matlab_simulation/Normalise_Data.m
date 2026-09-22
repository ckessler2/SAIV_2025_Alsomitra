clc; clear

script_dir = fileparts(mfilename('fullpath'));
data_dir = fullfile(script_dir, 'data');
python_data_dir = fullfile(script_dir, '..', 'python_training', 'data');

% load(fullfile(data_dir, "Normalisation_Constants.mat"))
% Cs = data2(:,1);
% Ss = data2(:,2);

% NOT NORMALIZED
load(fullfile(data_dir, 'Training_Data.mat'))
data = data3;

Cs = [];
Ss = [];
data_true = data;
data_norm = data;

% NORMALIZATION (gets Cs and Ss)
for i = 1:7
    [N,C,S] = normalize(data(:,i),"range") ;
    data_norm(:,i) = N;
    Cs = [Cs,C];
    Ss = [Ss,S];
end


writematrix(data_norm,fullfile(python_data_dir,'Training_Data_Normalised.csv')) 
save(fullfile(python_data_dir,'Training_Data_Normalised.mat'),'data_norm')
constants = [Cs; Ss];
save(fullfile(data_dir,'Normalisation_Constants.mat'),'constants')


data_true_2 = data_norm;

% CHECK METHOD

%  DENORMALIZATION (requires Cs and Ss)
for n = 1:length(data_norm)
    data_true_2(n,:) = (data_norm(n,:) .* Ss)+ Cs;
end

% NORMALIZATION (gets Cs and Ss), manually
data_norm_2 = data_true;
for n = 1:length(data_norm)
    % for n = 1:length(data_norm)
    %     data_norm_2(n,i) = (data_true(n,i) - Cs(i)) / Ss(i);
    % end
    data_norm_2(n,:) = (data_true(n,:) - Cs) ./ Ss;

end
