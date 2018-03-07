# -*- coding: utf-8 -*-
import numpy as np
import scipy.spatial as sc
import pandas as pd
import time

np.random.seed()

def Fold(length,fold):
    inds = np.arange(length)
    np.random.shuffle(inds)
    result = []
    step = int(length/fold)
    for i in range(fold):
        train = np.hstack((inds[:i*step],inds[(i+1)*step:]))
        test = inds[i*step:(i+1)*step]
        result.append((train,test))
    return result

def MSE(y,y_true):
    return np.sum((y - y_true)**2)/len(y)

class Regressor:
    scale = None
    model = None
    theta = None
    h = 0
    kernel = None
    fold = 0
    X_tr = None
    y_tr = None

    def __init__(self, model='lin',kernel=None,fold=5):
        self.model = model
        self.fold = fold
        if kernel == 'boxcar':
            self.kernel = lambda x : (x <= 1 and x >= 0)
        elif kernel == 'gauss':
            self.kernel = lambda x : np.sqrt(2/np.pi)*np.exp(-(x**2) / 2)
        elif kernel == 'epanech':
            self.kernel = lambda x : 3/2 * (1 - x**2) * (x <= 1 and x >= 0)

    def fit(self,X,y):
        if len(X.shape) == 1:
            X = np.array([[x] for x in X])
        self.scale = np.max(X,axis = 0)
        X = X / self.scale
        if self.model == 'lin':
            self.fit_lin(X,y)
        elif self.model == 'NW' or self.model == 'loclin':
            self.fit_NW(X,y)

    def predict(self,X, h=None):
        if len(X.shape) == 1:
            X = np.array([[x] for x in X])
        if h is None:
            h = self.h
            X = X / self.scale
        if self.model == 'lin':
            return np.dot(np.hstack((X,np.ones((X.shape[0],1)))),self.theta)
        elif self.model == 'NW':
            kernels = lambda x,y,h : self.kernel(sc.distance.euclidean(x,y)/h)
            predict_one = lambda x : np.sum(
                    np.apply_along_axis(kernels,1,self.X_tr,*(x,h))*self.y_tr)/np.sum(
                            np.apply_along_axis(kernels,1,self.X_tr,*(x,h)))
            return np.invert(np.isnan(np.apply_along_axis(predict_one,1,X)))*np.apply_along_axis(predict_one,1,X)
        elif self.model == 'loclin':
            X_ = np.hstack((self.X_tr,np.ones((self.X_tr.shape[0],1))))
            kernels = lambda x,y,h : self.kernel(sc.distance.euclidean(x,y)/h)
            weights = lambda x : np.apply_along_axis(kernels,1,self.X_tr,*(x,h))
            theta = lambda x : np.linalg.lstsq(
                    np.dot(X_.T*weights(x),X_),
                    np.dot(X_.T*weights(x),self.y_tr), None)[0]
            predict_one = lambda x : np.dot(np.hstack((x,np.ones(1))), theta(x))
            return np.apply_along_axis(predict_one,1,X)

    def fit_lin(self,X,y):
        X_ = np.hstack((X,np.ones((X.shape[0],1))))
        self.theta = np.linalg.lstsq(np.dot(X_.T,X_),np.dot(X_.T,y),None)[0]

    def fit_NW(self,X,y):
        self.h = (10/X.shape[0])**(1/X.shape[1])
        self.X_tr = X
        self.y_tr = y


data = pd.read_csv('winequality-red.csv')
X = np.array(data.drop(columns=['quality']))
y = np.array(data['quality'],dtype=float)
models = ['NW','loclin']
kernels = ['boxcar','gauss','epanech']
errors = []
times_fit = []
times_predict = []
N = 5

kaberne = Regressor(model='lin')
error = 0.
time_f = 0.
time_p = 0.
for train,test in Fold(X.shape[0],N):
    t1 = time.time()
    kaberne.fit(X[train],y[train])
    t2 = time.time()
    y_ = kaberne.predict(X[test])
    t3 = time.time()
    error += MSE(y_,y[test])
    time_f += (t2-t1)
    time_p += (t3-t2)
error /= N
time_f /= N
time_p /= N
errors.append(error)
times_fit.append(time_f)
times_predict.append(time_p)

for model in models:
    for kernel in kernels:
        kaberne = Regressor(model=model,kernel=kernel)
        error = 0.
        time_f = 0.
        time_p = 0.
        for train,test in Fold(X.shape[0],N):
            t1 = time.time()
            kaberne.fit(X[train],y[train])
            t2 = time.time()
            y_ = kaberne.predict(X[test])
            t3 = time.time()
            error += MSE(y_,y[test])
            time_f += (t2-t1)
            time_p += (t3-t2)
        error /= N
        time_f /= N
        time_p /= N
        errors.append(error)
        times_fit.append(time_f)
        times_predict.append(time_p)

print(errors,times_fit,times_predict)