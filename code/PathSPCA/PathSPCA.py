#!/usr/bin/env python
import numpy as np
from numpy import linalg as la

def PathSPCA(A, k):
    """
    Implementation of PathSPCA algorithm for sparse principal component analysis
    
    Parameters:
    A: data matrix (samples x features)
    k: target sparsity
    
    Returns:
    vars_list: explained variance for each sparsity level
    selected_indices: indices of selected features
    rhos: rho values list
    """
    # Get data matrix dimensions
    M, N = A.shape
    
    # Calculate variance of each feature
    As = ((A*A).sum(axis=0))
    
    # Select feature with maximum variance as starting point
    vmax = As.max()
    vp = As.argmax()
    subset = [vp]
    
    # Initialize result lists
    vars_list = []
    rhos = [(A[:, vp]*A[:, vp]).sum()]
    
    # Initialize temporary matrix
    Stemp = np.array([[rhos[0]]])
    
    # Greedy feature selection
    for i in range(1, k):
        # Calculate maximum eigenvalue and eigenvector of current subset
        lev, v = la.eig(Stemp)
        lev = np.real(lev)
        v = np.real(v)
        
        # Add current maximum eigenvalue to results
        vars_list.append(lev.max())
        
        # Find corresponding eigenvector
        vp = lev.argmax()
        
        # Calculate projection vector
        x = np.dot(A[:, subset], v[:, vp])
        x = x / la.norm(x)
        
        # Find remaining features
        seto = list(range(N))
        for j in subset:
            if j in seto:
                seto.remove(j)
        
        # Calculate projection values for each remaining feature
        vals = np.zeros(len(seto))
        for j in range(len(seto)):
            vals[j] = np.dot(x.T, A[:, seto[j]]) ** 2
        
        # Select feature with maximum projection value
        rhos.append(vals.max())
        vpo = seto[np.argmax(vals)]
        
        # Update temporary matrix
        new_col = np.dot(A[:, subset].T, A[:, vpo]).reshape(-1, 1)
        Stemp = np.hstack((Stemp, new_col))
        
        new_row = np.append(np.dot(A[:, vpo].T, A[:, subset]), [(A[:, vpo]*A[:, vpo]).sum()])
        Stemp = np.vstack((Stemp, new_row))
        
        # Add selected feature to subset
        subset.append(vpo)
    
    # Calculate final eigenvalue
    lev, v = la.eig(Stemp)
    vars_list.append(np.real(lev).max())
    
    return vars_list, subset, rhos

if __name__ == "__main__":
    # Test code
    k = 2  # Target sparsity
    A = np.array([[1, 2, 3, 4, 3, 2, 1],
               [4, 2, 1, 4, 3, 2, 1],
               [5, 2, 3, 4, 3, 3, 1]])

    # Call function
    vars_list, selected_indices, rhos = PathSPCA(A, k)
    print("Selected indices:", selected_indices)
    print("Explained variances:", vars_list)
    print("Rho values:", rhos)