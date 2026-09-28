Alternative (naive, non-adversarial) training route using the orange data mining toolkit. 

Import the training csv on the left, and set column 6 as the desired output using "Select Columns". 

This labelled data goes into a NN node (6/4 hidden layers, ReLu activation, L-BFGS-B Solver, alpha = 0.06, 200 iterations).

NN performance is evaluated using "Test and Score" (R2 = 0.995), and "Predictions - Scatter plot".

![My diagram](https://github.com/ckessler2/SAIV_2025_Alsomitra/blob/main/python_training/Orange/Alsomitra_improved.svg)

