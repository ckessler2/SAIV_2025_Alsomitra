## Alternative (naive regression) training route using Orange 3, a visual programming environment. 

Import the training csv on the left, and set column 6 as the desired output using "Select Columns". Labelled data goes into a NN node (6/4 hidden layers, ReLu activation, L-BFGS-B Solver, alpha = 0.06, 200 iterations). Performance is evaluated using "Test and Score" (R2 = 0.995), and "Predictions - Scatter plot" (plot 6th column vs NN output).

![My diagram](https://github.com/ckessler2/SAIV_2025_Alsomitra/blob/main/python_training/Orange/Alsomitra_improved.svg)

<p align="center">
  <img src="https://github.com/ckessler2/SAIV_2025_Alsomitra/blob/main/python_training/Orange/regression_performance.png" width="33%" alt="Description">
</p>
