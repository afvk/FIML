# Field inversion and machine learning


## Introduction

Machine learning has a huge impact in fields where theory-based models have failed to perform in a satisfactory manner. Examples of these fields are image processing, speech recognition, and machine translation. However, machine learning can also play an important role in improving methods in fields where physical theories have dominated. An important difference with aforementioned fields is that in physics dominated domains, the majority of the problem can be modeled using physical laws. Directly applying machine learning to map the input of the problem to some output of interest usually does not work that well. However, finding the step where the largest assumption is made and applying machine learning to this step can significantly improve the results of these methods. 


## Field inversion and machine learning

One way to implement machine learning to physical models is the paradigm of field inversion and machine learning [1]. An important advantage of this paradigm is that enables incorporating prior knowledge into the model, and also allows the modeler to extract modeling knowledge from the results. The paradigm can be summarized as follows:

1. Define some corrective term in the base model
2. Extract the optimal corrective function from high fidelity data
3. Train a machine learning model to estimate the corrective function, given a set of features


### Optimization problem

One way of defining the optimization step is in maximizing the probability of the corrective term given the data, i.e. finding the maximum a posteriori (MAP) solution. Assuming that the prior and the discrepancy between the model output and the high-fidelity data are normally distributed gives us the following posterior.

![](https://latex.codecogs.com/svg.latex?-%20%5Clog%20p%20%28%20%5Cboldsymbol%7B%5Cbeta%7D%20%7C%20%5Cmathbf%7Bd%7D%20%29%20%3D%20%5Cunderbrace%7B%20%5Cfrac%7B1%7D%7B2%7D%20%28%20h%28%5Cboldsymbol%7B%5Cbeta%7D%29%20-%20%5Cmathbf%7Bd%7D%29%5ET%20C_m%5E%7B-1%7D%20%28%20h%28%5Cboldsymbol%7B%5Cbeta%7D%29%20-%20%5Cmathbf%7Bd%7D%29%20&plus;%20%5Cfrac%7B1%7D%7B2%7D%20%28%20%5Cboldsymbol%7B%5Cbeta%7D%20-%20%5Cboldsymbol%7B%5Cbeta%7D_%5Ctext%7Bprior%7D%29%20%29%5ET%20C_%5Cbeta%5E%7B-1%7D%20%28%20%5Cboldsymbol%7B%5Cbeta%7D%20-%20%5Cboldsymbol%7B%5Cbeta%7D_%5Ctext%7Bprior%7D%29%20%29%7D_J)

As we want to maximize the posterior, our optimization objective is to minimize $J$. If we choose our prior and observational covariance matrices to be simple identity matrices multiplied with some constant, the optimization problem reduces to minimizing the sum of the square discrepancies with a regularization term. 

### Gradients

If we want to use gradient-based optimization methods, we need some way to find the gradient of the objective function with respect to the corrective term. For small scale problems, it is easy to find these gradient using a finite difference approximation. However, in applications where the modeling problem is discretized into a large number of cells (e.g. in computational fluid dynamics), this approach is computationally unfeasible. 


We can rewrite the gradient of the objective function using the chain rule. 

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%20J%7D%7Bd%20%5Cbeta%7D%20%3D%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Cbeta%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20T%7D%20%5Cfrac%7Bd%20T%7D%7Bd%20%5Cbeta%7D)

The explicit derivatives are easy to obtain: they can be derived directly from our definition of the objective function. However, the sensitivity of our variables cannot be obtained straightforwardly. Also, we have a set of governing equations which we can rewrite as

![](https://latex.codecogs.com/svg.latex?R%20%28T%2C%20%5Cbeta%29%20%3D%200)

As we don't want the validity of our governing equations to change if we change the corrective term, we can write

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%20R%7D%7Bd%20%5Cbeta%7D%20%3D%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20%5Cbeta%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20T%7D%20%5Cfrac%7Bd%20T%7D%7Bd%20%5Cbeta%7D%20%3D%200)

Again, the explicit derivatives follow straightforwardly from the discretization of the governing equations. Introducing some new set of variables $\psi$, which we will determine later, we can write the gradient as

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%20J%7D%7Bd%20%5Cbeta%7D%20%3D%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Cbeta%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20T%7D%20%5Cfrac%7Bd%20T%7D%7Bd%20%5Cbeta%7D%20&plus;%20%5Cpsi%5ET%20%5Cleft%28%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20%5Cbeta%7D%20&plus;%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20T%7D%20%5Cfrac%7Bd%20T%7D%7Bd%20%5Cbeta%7D%20%5Cright%29%20%3D%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Cbeta%7D%20&plus;%20%5Cpsi%5ET%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20%5Cbeta%7D%20&plus;%20%5Cunderbrace%7B%20%5Cleft%28%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20T%7D%20&plus;%20%5Cpsi%5ET%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20T%7D%20%5Cright%29%20%7D_%5Ctext%7Bset%20to%20zero%7D%20%5Cfrac%7Bd%20T%7D%7Bd%20%5Cbeta%7D)

We will call this new set of variables the adjoint variables. Using the constraint that the indicated term should be zero, they can be determined by solving a system of linear equations. 

![](https://latex.codecogs.com/svg.latex?%5Cleft%28%20%5Cfrac%7B%5Cpartial%20R%7D%7B%5Cpartial%20T%7D%20%5Cright%29%5ET%20%5Cpsi%20%3D%20-%20%5Cleft%28%20%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20T%7D%20%5Cright%29%5ET)

We now have an expression for the gradients which we can easily evaluate, given that we do one extra system solve. Note that our gradient calculation is now practically independent of the number of points in our simulation. 


## Example

To illustrate the paradigm, [1] uses the following scalar ordinary differential equation

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%5E2%20T%7D%7Bdz%5E2%7D%20%3D%20%5Cvarepsilon%28T%29%28T_%5Cinfty%5E4%20-%20T%5E4%29%20&plus;%20h%28T_%5Cinfty%20-%20T%29)

where ![](https://latex.codecogs.com/svg.latex?%5Cinline%20z%20%5Cin%20%5B0%2C1%5D%24%2C%20%24T_%5Cinfty) can be a function of z, T is our primal variable, and 

![](https://latex.codecogs.com/svg.latex?%5Cvarepsilon%20%28T%29%20%3D%20%5Cleft%281%20&plus;%205%20%5Csin%20%5Cleft%28%20%5Cfrac%7B3%20%5Cpi%20T%7D%7B200%7D%20&plus;%20e%5E%7B0.02%20T%7D%20&plus;%20%5Cmathcal%7BN%7D%28%200%2C%200.1%5E2%20%5Cright%29%20%5Cright%29%20%5Ccdot%2010%5E%7B-4%7D)

where h = 0.5. Let's say we want to model this process using 

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%5E2%20T%7D%7Bdz%5E2%7D%20%3D%20%5Cvarepsilon_0%20%28T_%5Cinfty%5E4%20-%20T%5E4%29)

and want to enhance this model using a spatially varying corrective term, 

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7Bd%5E2%20T%7D%7Bdz%5E2%7D%20%3D%20%5Cbeta%20%28z%29%20%5Cvarepsilon_0%20%28T_%5Cinfty%5E4%20-%20T%5E4%29)

The convenience of illustrating the paradigm using a simple model problem like this is that we can derive the true form of the corrective term.

![](https://latex.codecogs.com/svg.latex?%5Cbeta_%5Ctext%7Btrue%7D%20%3D%20%5Cfrac%7B1%7D%7B%5Cvarepsilon_0%7D%20%5Cleft%28%201%20&plus;%205%20%5Csin%20%5Cleft%28%20%5Cfrac%7B3%20%5Cpi%20T%7D%7B200%7D%20&plus;%20e%5E%7B0.02%20T%7D%20&plus;%20%5Cmathcal%7BN%7D%28%200%2C%200.1%5E2%20%5Cright%29%20%5Cright%29%20%5Ccdot%2010%5E%7B-4%7D%20&plus;%20%5Cfrac%7Bh%7D%7B%5Cvarepsilon_0%7D%20%5Cfrac%7BT_%5Cinfty%20-%20T%7D%7BT_%5Cinfty%5E4%20-%20T%5E4%7D)



### Discretization

#### Forward problem/primal equation

The problem can be discretized using finite volume discretization with homogeneous boundary conditions. Using a central difference scheme for the second order derivative and rewriting the equation for the temperature in cell i gives

![](https://latex.codecogs.com/svg.latex?T_i%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%28%20T_%7Bi-1%7D%20&plus;%20T_%7Bi&plus;1%7D%20-%20%5Cleft%28%20%5CDelta%20z%20%5Cright%29%5E2%20%5Cleft%28%20%5Cvarepsilon%20%28T_i%29%20%5Cleft%28T_i%5E4%20-%20T_%7B%5Cinfty%2Ci%7D%5E4%20%5Cright%29%20&plus;%20h%20%28T_i%20-%20T_%7B%5Cinfty%2Ci%7D%29%20%5Cright%29%20%5Cright%29)

Similarly, the base model and the augmented model can be solved as

![](https://latex.codecogs.com/svg.latex?T_i%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%28%20T_%7Bi-1%7D%20&plus;%20T_%7Bi&plus;1%7D%20-%20%5Cleft%28%20%5CDelta%20z%20%5Cright%29%5E2%20%5Cvarepsilon_0%20%5Cleft%28T_i%5E4%20-%20T_%7B%5Cinfty%2Ci%7D%5E4%20%5Cright%29%20%5Cright%29)

and

![](https://latex.codecogs.com/svg.latex?T_i%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%28%20T_%7Bi-1%7D%20&plus;%20T_%7Bi&plus;1%7D%20-%20%5Cleft%28%20%5CDelta%20z%20%5Cright%29%5E2%20%5Cbeta_i%20%5Cvarepsilon_0%20%5Cleft%28T_i%5E4%20-%20T_%7B%5Cinfty%2Ci%7D%5E4%20%5Cright%29%20%5Cright%29)

These equations are then solved iteratively until convergence, using under-relaxation to stabilize the iterations, i.e.

![](https://latex.codecogs.com/svg.latex?T_i%5En%20%5Cleftarrow%20%5Calpha%20T_i%5En%20&plus;%20%281-%5Calpha%20%29%20T_i%5E%7Bn-1%7D)

where ![](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Calpha%20%5Cin%20%5B0%2C1%5D) trades off stability (for low alpha) and convergence (for high alpha). The iterations are stopped once the L2-norm between two consecutive solutions drops below a specified criterion. 


#### Adjoint equation

The partial derivatives necessary for setting up the adjoint equation require taking two scalar-by-vector and two vector-by-vector derivatives of the objective function and the governing equation, respectively. This can be done conveniently using Einstein summation convention. 

![](https://latex.codecogs.com/svg.latex?J%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%5Cleft%5B%20%28%20H_%7Bij%7D%20T_j%20-%20d_i%29%20%28C_m%5E%7B-1%7D%29_%7Bik%7D%20%28%20H_%7Bkl%7D%20T_l%20-%20d_k%29%20&plus;%20%28%20%5Cbeta_i%20-%20%5Cbeta_%7B%5Ctext%7Bprior%7D%2Ci%7D%20%29%20%29%20%28C_%5Cbeta%5E%7B-1%7D%29_%7Bij%7D%20%28%20%5Cbeta_j%20-%20%5Cbeta_%7B%5Ctext%7Bprior%7D%2Cj%7D%20%29%20%5Cright%5D)

Making use of the fact that the prior and observational covariance matrices are symmetric, and using 

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20T_i%7D%7B%5Cpartial%20T_j%7D%20%3D%20%5Cfrac%7B%5Cpartial%20%5Cbeta_i%7D%7B%5Cpartial%20%5Cbeta_j%7D%20%3D%20%5Cdelta_%7Bij%7D)

where ![](https://latex.codecogs.com/svg.latex?%5Cinline%20%5Cdelta_%7Bij%7D) is the Kronecker delta, we can easily derive 

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20%5Cbeta_k%7D%20%3D%20%28C_%5Cbeta%5E%7B-1%7D%29_%7Bkj%7D%20%28%20%5Cbeta_j%20-%20%5Cbeta_%7B%5Ctext%7Bprior%7D%2Cj%7D%20%29)

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20J%7D%7B%5Cpartial%20T_m%7D%20%3D%20%28H_%7Bkl%7D%20T_l%20-%20d_k%29%20%28C_m%5E%7B-1%7D%29_%7Bik%7D%20H_%7Bim%7D)

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20R_%5Calpha%7D%7B%5Cpartial%20%5Cbeta_j%7D%20%3D%20%5Cfrac%7B1%7D%7B2%7D%20%28%5CDelta%20z%29%5E2%20%5Cdelta_%7B%5Calpha%20j%7D%20%5Cvarepsilon_0%20%28T_%5Calpha%5E4%20-%20T_%7B%5Cinfty%2C%20%5Calpha%7D%5E4%20%29)

![](https://latex.codecogs.com/svg.latex?%5Cfrac%7B%5Cpartial%20R_%5Calpha%7D%7B%5Cpartial%20T_j%7D%20%3D%20%5Cdelta_%7B%5Calpha%20_j%7D%20%5Cleft%28%201%20&plus;%202%20T_%5Calpha%5E3%20%28%5CDelta%20z%29%5E2%20%5Cbeta_%5Calpha%20%5Cvarepsilon_0%20%5Cright%29%20-%20%5Cfrac%7B1%7D%7B2%7D%20%5Cdelta_%7B%5Calpha-1%2Cj%7D%20-%20%5Cfrac%7B1%7D%7B2%7D%20%5Cdelta_%7B%5Calpha&plus;1%2Cj%7D)




[1] Parish, E. J., & Duraisamy, K. (2016). A paradigm for data-driven predictive modeling using field inversion and machine learning. Journal of Computational Physics, 305, 758-774.