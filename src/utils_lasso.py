import numpy as np 
from scipy.stats import t,invgamma


def update_parameters(y_tilde, X, sigma2_prev, tau2_prev, lambda_sq):
    """
    Algorithm 1: Bayesian Lasso Transition Kernel
    """
    n, p = X.shape
    
    # ---------------------------------------------------------
    # 1. Update beta
    # ---------------------------------------------------------

    D_tau_inv = np.diag(1.0 / tau2_prev)
    
    # A = X^T X + D_tau^(-1)
    XtX = X.T @ X
    A = XtX + D_tau_inv
    

    # A * mu_beta = X^T * y_tilde  =>  mu_beta = A^(-1) * X^T * y_tilde
    Xt_y = X.T @ y_tilde
    mu_beta = np.linalg.solve(A, Xt_y)
    
    # Sigma_beta = sigma^2 * A^(-1)
    A_inv = np.linalg.inv(A)
    Sigma_beta = sigma2_prev * A_inv
    
    # Sample beta^(t)
    beta_t = np.random.multivariate_normal(mu_beta, Sigma_beta)
    
    # ---------------------------------------------------------
    # 2. Update sigma^2
    # ---------------------------------------------------------
    residuals = y_tilde - X @ beta_t
    RSS = residuals.T @ residuals
    
    shape_sigma = (n - 1 + p) / 2.0
    # beta^T * D_tau^(-1) * beta
    penalty_term = beta_t.T @ D_tau_inv @ beta_t
    scale_sigma = (RSS + penalty_term) / 2.0
    
    # Sample sigma^{2, (t)}
    sigma2_t = invgamma.rvs(a=shape_sigma, scale=scale_sigma)
    
    # ---------------------------------------------------------
    # 3. Update tau^2
    # ---------------------------------------------------------
    tau2_t = np.zeros(p)
    
    for j in range(p):
        # Safeguard against division by exactly zero if beta is extremely small
        beta_j_sq = max(beta_t[j]**2, 1e-10)
        
        # mu = sqrt( (lambda^2 * sigma^2) / beta_j^2 )
        mu_invgauss = np.sqrt((lambda_sq * sigma2_t) / beta_j_sq)
        
        # Sample 1/tau^2 from Inverse-Gaussian (Wald distribution)
        inv_tau2_j = np.random.wald(mean=mu_invgauss, scale=lambda_sq)
        tau2_t[j] = 1.0 / inv_tau2_j
        
    return beta_t, sigma2_t, tau2_t



# Standarize X and y 
def standarized_centered_X_y(X,y):
    X_covariates_only = X[:, 1:] 

    # Now run the exact same standardizing code on this new matrix!
    X_mean = np.mean(X_covariates_only, axis=0)
    X_std = np.std(X_covariates_only, axis=0)

    X_std[X_std == 0] = 1.0 

    X_stdized = (X_covariates_only - X_mean) / X_std


    y_mean = np.mean(y)
    y_tilde = y - y_mean

    return X_stdized,y_tilde


import numpy as np
from scipy.stats import invgamma

def run_gibbs_fixed_lambda(y_tilde, X, lambda_fixed, T=10000, B=1000,random_seed=2026):
    np.random.seed(random_seed)
    """
    Gibbs Sampler for Bayesian Lasso with a fixed penalty parameter lambda.
    
    Parameters:
    -----------
    y_tilde   : Centered response vector (n,)
    X         : Standardized design matrix (n, p)
    lambda_fixed : The fixed lambda value (e.g., from Empirical Bayes Phase 1)
    T         : Total number of iterations
    B         : Burn-in iterations
    
    Returns:
    --------
    results : dict containing posterior samples of beta, sigma2, and tau2
    """
    n, p = X.shape
    lambda_sq = lambda_fixed**2
    
    # 1. Initialize parameters
    # Starting with OLS-like estimates or zeros
    beta_curr = np.zeros(p)
    sigma2_curr = 1.0
    tau2_curr = np.ones(p) # Critical to start > 0 for D_tau_inv
    
    # 2. Storage for samples (only storing T-B samples)
    num_samples = T - B
    beta_samples = np.zeros((num_samples, p))
    sigma2_samples = np.zeros(num_samples)
    tau2_samples = np.zeros((num_samples, p))
    
    print(f"Running Gibbs Sampler (T={T}, B={B}) with fixed λ={lambda_fixed:.4f}...")
    
    for t in range(T):
        # Apply the transition kernel (your update_parameters function)
        beta_curr, sigma2_curr, tau2_curr = update_parameters(
            y_tilde, X, beta_curr, sigma2_curr, tau2_curr, lambda_sq
        )
        
        # 3. Store results only after burn-in
        if t >= B:
            idx = t - B
            beta_samples[idx, :] = beta_curr
            sigma2_samples[idx] = sigma2_curr
            tau2_samples[idx, :] = tau2_curr
            
    return {
        'beta': beta_samples,
        'sigma2': sigma2_samples,
        'tau2': tau2_samples
    }