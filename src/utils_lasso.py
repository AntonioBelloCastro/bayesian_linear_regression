import numpy as np
from scipy.stats import invgamma, gamma


def update_parameters(y_tilde, X, sigma2_prev, tau2_prev, lambda_sq):
    """
    Algorithm 1: Bayesian Lasso Transition Kernel
    """
    n, p = X.shape
    
    # ---------------------------------------------------------
    # 1. Update beta

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
    tau2_t = np.zeros(p)
    
    for j in range(p):
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

    X_mean = np.mean(X_covariates_only, axis=0)
    X_std = np.std(X_covariates_only, axis=0)

    X_std[X_std == 0] = 1.0 

    X_stdized = (X_covariates_only - X_mean) / X_std

    y_mean = np.mean(y)
    y_tilde = y - y_mean

    return X_stdized,y_tilde




def bayesian_lasso_hyp(y_tilde, X, r=1.0, delta=0.1, T=11000, B=1000,random_seed=2026):
    """
    Algorithm 3: Bayesian Lasso Gibbs Sampler with lambda hyperprior
    t=0 Casella Initialisation
    """
    np.random.seed(random_seed)
    n, p = X.shape
    
    # Arrays to store the values of each parameter for each iteration
    beta_samples = np.zeros((T, p))
    sigma2_samples = np.zeros(T)
    tau2_samples = np.zeros((T, p))
    lambda_sq_samples = np.zeros(T) 
    
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_curr = XtX_inv @ X.T @ y_tilde
    
    residuals_ols = y_tilde - X @ beta_curr
    sigma2_curr = (residuals_ols.T @ residuals_ols) / (n - p - 1)
    
    sum_abs_beta = np.sum(np.abs(beta_curr))
    lambda_curr = (p * np.sqrt(sigma2_curr)) / sum_abs_beta
    lambda_sq_curr = lambda_curr**2
    
    tau2_curr = np.zeros(p)
    for j in range(p):
        beta_j_sq = max(beta_curr[j]**2, 1e-10)
        mu_invgauss = np.sqrt((lambda_sq_curr * sigma2_curr) / beta_j_sq)
        inv_tau2_j = np.random.wald(mean=mu_invgauss, scale=lambda_sq_curr)
        tau2_curr[j] = 1.0 / inv_tau2_j

    for t in range(T):
        
        beta_curr, sigma2_curr, tau2_curr = update_parameters(
            y_tilde, X, sigma2_curr, tau2_curr, lambda_sq_curr
        )
        
        shape_param = p + r
        rate_param = 0.5 * np.sum(tau2_curr) + delta
        scale_param = 1.0 / rate_param
        lambda_sq_curr = gamma.rvs(a=shape_param, scale=scale_param)
        
        beta_samples[t, :] = beta_curr
        sigma2_samples[t] = sigma2_curr
        tau2_samples[t, :] = tau2_curr
        lambda_sq_samples[t] = lambda_sq_curr
        
    return {
        'beta': beta_samples[B:],
        'sigma2': sigma2_samples[B:],
        'tau2': tau2_samples[B:],
        'lambda_sq': lambda_sq_samples[B:]
    }


def bayesian_lasso_em(y_tilde, X, epsilon=1e-6, M=50, K=200, T=11000, B=1000,random_seed=2026):
    np.random.seed(random_seed)
    """
    Algorithm 2: Bayesian Lasso with Pure Monte Carlo EM for lambda
    """
    n, p = X.shape
    
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta_curr = XtX_inv @ X.T @ y_tilde
    
    residuals_ols = y_tilde - X @ beta_curr
    sigma2_curr = (residuals_ols.T @ residuals_ols) / (n - p - 1)
    
    sum_abs_beta = np.sum(np.abs(beta_curr))
    lambda_curr = (p * np.sqrt(sigma2_curr)) / sum_abs_beta
    lambda_sq_curr = lambda_curr**2
    
    tau2_curr = np.zeros(p)
    for j in range(p):
        beta_j_sq = max(beta_curr[j]**2, 1e-10)
        mu_invgauss = np.sqrt((lambda_sq_curr * sigma2_curr) / beta_j_sq)
        inv_tau2_j = np.random.wald(mean=mu_invgauss, scale=lambda_sq_curr)
        tau2_curr[j] = 1.0 / inv_tau2_j

    lambda_history = [lambda_curr]
    
    for k in range(1, K + 1):
        
        tau2_samples_E_step = np.zeros((M, p))
        
        for m in range(M):
            beta_curr, sigma2_curr, tau2_curr = update_parameters(
                y_tilde, X, sigma2_curr, tau2_curr, lambda_sq_curr
            )
            tau2_samples_E_step[m, :] = tau2_curr
            
        E_tau2 = np.mean(tau2_samples_E_step, axis=0)
        lambda_next = np.sqrt((2.0 * p) / np.sum(E_tau2))
        
        if abs(lambda_next - lambda_curr) < epsilon:
            lambda_curr = lambda_next
            lambda_history.append(lambda_curr)
            print(f" EM Converged at iteration k={k} to lambda={lambda_curr:.4f}")
            break
            
        lambda_curr = lambda_next
        lambda_sq_curr = lambda_curr**2
        lambda_history.append(lambda_curr)
        

    # Final lambda estimate
    lambda_eb = lambda_curr
    lambda_sq_eb = lambda_eb**2

    # ------------------------------------------------
    # Phase 2: Final Gibbs Sampler with fixed lambda
    print(f"Starting Phase 2: Posterior Inference (T={T} iterations)...")
    
    # Arrays to store final posterior samples
    beta_samples = np.zeros((T, p))
    sigma2_samples = np.zeros(T)
    tau2_samples = np.zeros((T, p))
    
    
    for t in range(T):
        beta_curr, sigma2_curr, tau2_curr = update_parameters(
            y_tilde, X, sigma2_curr, tau2_curr, lambda_sq_eb
        )
        
        beta_samples[t, :] = beta_curr
        sigma2_samples[t] = sigma2_curr
        tau2_samples[t, :] = tau2_curr

    # Return post burn-in samples
    return {
        'lambda_EB': lambda_eb,
        'lambda_history': np.array(lambda_history),
        'beta_samples': beta_samples[B:],   
        'sigma2_samples': sigma2_samples[B:],
        'tau2_samples': tau2_samples[B:]
    }