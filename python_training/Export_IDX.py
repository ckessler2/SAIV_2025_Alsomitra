# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 10:28:10 2025

@author: Colin Kessler
"""
from data_handler import DataHandler  # Handles data loading and preprocessing
import pandas as pd
import numpy as np
import idx2numpy

filepath = "Training_Data_Normalised.csv"  # Path to the dataset file
target_column = "target"  # Column name for the labels in the dataset
input_size = 6  # Number of input features for the model

data = pd.read_csv(filepath, delim_whitespace=False, header=None)
data.columns = [
    "dv_xpdt",
    "dv_ypdt",
    "domegadt",
    "dthetadt",
    "dx_dt",
    "dy_dt",
    "target",
]

X = data.drop(target_column, axis=1)
y = data[target_column]

# Double data to match doubled networks
# X = pd.concat([X, X], axis=1)
# y = pd.concat([y, y], axis=1)

X_numpy = X.to_numpy().astype("float32")
y_numpy = y.to_numpy().astype("float32")

file_path1 = "dataset_inputs_norm.idx"
file_path2 = "dataset_outputs_norm.idx"

f_write = open(file_path1, "wb")  # Open file in write-binary mode
idx2numpy.convert_to_file(f_write, X_numpy)
f_write.close()  

f_write = open(file_path2, "wb")  # Open file in write-binary mode
idx2numpy.convert_to_file(f_write, y_numpy)
f_write.close()  