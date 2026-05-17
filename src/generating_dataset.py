import numpy as np

def data_generator(true_beta,n=100,random_seed=2026):
    np.random.seed(random_seed)
    n = n
    p = len(true_beta)-1

    Sigma = np.eye(p) #Corr Matrix

    
    X_raw = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=n)
    X_std = (X_raw - np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0) 

    
    X = np.c_[np.ones(n), X_std] #Intercept

    
    error = np.random.normal(0, 2.0, n) #Random error
    y = X @ true_beta + error

    return (X,y)

true_beta = np.array([-1, 2.0,4.0,-1.5,0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
dataset = data_generator(true_beta)
dataset_10000=data_generator(true_beta,n=10000)