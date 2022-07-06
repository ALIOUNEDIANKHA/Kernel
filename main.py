#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score ,classification_report
from sklearn.preprocessing import StandardScaler

from matplotlib import pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


def rbf_kernel(X1, X2, sigma=None):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    if sigma is None:
            sigma = sigma_from_median(X)
    # For loop with rbf_kernel_element works but is slow in python
    # Use matrix operations!
    X2_norm = np.sum(X2 ** 2, axis=-1)
    X1_norm = np.sum(X1 ** 2, axis=-1)
    gamma = 1 / (2 * sigma ** 2)
    K = np.exp(- gamma * (X1_norm[:, None] + X2_norm[None, :] - 2 * np.dot(X1, X2.T)))
    return K

def linear_kernel(X1, X2):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the linear kernel
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    '''
    return X1.dot(X2.T)

def polynomial_kernel(X1, X2, degree=3):
    '''
    Returns the kernel matrix K(X1_i, X2_j): size (n1, n2)
    where K is the polynomial kernel of degree `degree`
    
    Input:
    ------
    X1: an (n1, p) matrix
    X2: an (n2, p) matrix
    degree: int
    '''
    return (1 + linear_kernel(X1, X2))**degree


# In[ ]:


def sigma_from_median(X):
    '''
    Returns the median of ||Xi-Xj||
    
    Input
    -----
    X: (n, p) matrix
    '''
    pairwise_diff = X[:, :, None] - X[:, :, None].T
    pairwise_diff *= pairwise_diff
    euclidean_dist = np.sqrt(pairwise_diff.sum(axis=1))
    return np.median(euclidean_dist)


# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class KernelLogisticRegression():
    '''
    Kernel Logistic Regression
    
    Methods
    ----
    fit
    predict
    '''
    kernels_ = {
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'rbf': rbf_kernel,
        # 'custom_kernel': custom_kernel, # Your kernel
    }
    def __init__(self, lambd=0.1, kernel='linear', **kwargs):
        self.lambd = lambd
        self.kernel_name = kernel
        self.kernel_function_ = self.kernels_[kernel]
        self.kernel_parameters = self.get_kernel_parameters(**kwargs)
        
        
    def get_kernel_parameters(self, **kwargs):
        params = {}
        if self.kernel_name == 'rbf':
            params['sigma'] = None
        if self.kernel_name == 'polynomial':
            params['degree'] = kwargs.get('degree', 2)
        # if self.kernel_name == 'custom_kernel':
        #     params['parameter_1'] = kwargs.get('parameter_1', None)
        #     params['parameter_2'] = kwargs.get('parameter_2', None)
        return params

    def WKRR(self, K, y, sample_weights=None):
        '''
        Weighted Kernel Ridge Regression

        This is just used for the KernelLogistic following up
        '''

        self.y_train = y
        n = len(self.y_train)

        w = np.ones_like(self.y_train) if sample_weights is None else sample_weights
        W = np.diagflat(np.sqrt(w))

        A = W.dot(K).dot(W)
        A[np.diag_indices_from(A)] += self.lambd * n
        # self.alpha = W (K + n lambda I)^-1 W y
        return W.dot(np.linalg.solve(A , W.dot(self.y_train)))

    def fit(self, X, y, max_iter=100, tol=1e-5):
    
        self.X_train = X
        self.y_train = y
        
        K = self.kernel_function_(X, X, **self.kernel_parameters)
        
        

        # Initialize
        alpha = np.zeros_like(self.y_train)
        # Iterate until convergence or max iterations
        for n_iter in range(max_iter):
            alpha_old = alpha
            f = K.dot(alpha_old)
            w = sigmoid(f) * sigmoid(-f)
            z = f + y / sigmoid(-y*f)
            # IRLS
            alpha = self.WKRR(K, z, sample_weights=w)
            # Break condition (achieved convergence)
            if np.sum((alpha-alpha_old)**2) < tol:
                break
        self.n_iter = n_iter
        self.alpha = alpha

        return self
            
        
    def decision_function(self, X):
        K_x = self.kernel_function_(X, self.X_train, **self.kernel_parameters)    
        return K_x

    def predict(self, X):
        return np.sign(self.decision_function(X) @ self.alpha)


# In[ ]:


X_train = pd.read_csv('Xtr_vectors.csv')
X_train.drop('Id', inplace=True, axis=1)
y_train = pd.read_csv('Ytr.csv')
y_train.drop('Id', inplace=True, axis=1)
X_test = pd.read_csv('Xte_vectors.csv')
iD = pd.DataFrame(X_test.Id)
X_test.drop('Id', inplace=True, axis=1)


# In[ ]:


X = np.array(X_train)

y = np.array(y_train)
Xt = np.array(X_test)
y = 2*y - 1

sc = StandardScaler().fit(X)
X = sc.transform(X)
Xt = sc.transform(Xt)


# In[ ]:


def submission():   
    best_lambd = 5.
    kernel = 'rbf'
    degree = 2
    sigma = None
    # Parameters already specified in the tune function
    model = KernelLogisticRegression(lambd=best_lambd, kernel=kernel, sigma=sigma, degree=degree)
    y_pred = model.fit(X, y).predict(Xt)
    #data frame for prediction
    pred = pd.DataFrame(y_pred>0, columns = ["Covid"])
    #stack the data frame for id and pred
    result = pd.DataFrame(np.hstack([iD,pred]), columns = ["ID", "Covid"])
    result.to_csv("Yte.csv",index=False)


# In[ ]:




