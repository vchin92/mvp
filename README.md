# Efficient data augmentation for multivariate probit models with panel data: An application to general practitioner decision-making about contraceptives
The code here provides a method for sampling the potentially high dimensional correlation matrix in a multivariate probit model for longitudinal data within a Markov chain Monte Carlo algorithm. In addition, antithetic sampling is performed to lower the autocorrelation between the iterates of random effects and regression coefficients.

### Getting started
Save all the files in a folder and change MATLAB's current directory to the folder that contains all the saved files.

The main program is `example.m` which loads the simulated data in `simulated_data.mat` and samples all parameters from their posterior distributions.

To run the program, type `example` in the command window and press enter. 

### Author
Vincent Chin (email: <vincent.chin@sydney.edu.au>)

### License
There are very few restrictions on the use of the codes - see the LICENSE.md file for details.

### Acknowledgement
The code for the functions `NUTS.m` and `dualAveraging.m` is based on the implementation by Matthew D. Hoffman (2011).

### Reference
Chin, V., D. Gunawan, D. G. Fiebig, R. Kohn, and S. A. Sisson (2020). Efficient data augmentation for multivariate probit models with panel data: an application to general practitioner decision-making about contraceptives. *Journal of the Royal Statistical Society: Series C (Applied Statistics) 69*(2), 277–300.
