#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 11 11:34:36 2023

@author: erkuttekeli
"""

### RUN for Monte Carlo Simulation######
### > sim(100)
###
### RUN for Diabetes dataset######
### > diabetes()
###
### RUN for watermelon dataset######
### > watermelon()


import numpy as np
from Qsun.Qcircuit import *
from Qsun.Qgates import *
from Qsun.Qmeas import *

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error
from numpy.linalg import eig
from numpy.linalg import inv
#import time
import math
import csv

def kappa(kor):
    w,_ = eig(kor)
    if max(w)==0:
        kp = 0
    else:
        kp = np.sqrt(max(w)/min(w))
    return kp

def vif(kort):
    return max(np.diagonal(kort))
    

def artificial_data(beta, rho, sigma, n, p1, p2):
    p = p1+p2
    omega = np.zeros((n,p))
    for i in range(p):
        omega[:,i] = np.random.normal(size=(n,))
    omega_cor = np.random.normal(size=(n,)) 
    adata = omega
    for i in range(p1):
       #adata[:,i] = np.sqrt(1-rho*rho)*omega[:,i] + rho*omega_cor
       adata[:,i] = (1-rho)*omega[:,i] + rho*omega_cor
    err = np.random.normal(scale=sigma, size=(n,))
    y = np.matmul(adata, np.transpose(beta)) + err
    return( adata, y )

def circuit(params):
    c = Qubit(1)
    RX(c, 0, params[0])
    RY(c, 0, params[1])
    return c

def output(params):
    c = circuit(params)
    prob = c.probabilities()
    return 1*prob[0] - 1*prob[1]

def outputM(params):
    ou = np.zeros(nc)
    for j in range(nc):
        ou[j] = output(params[j,:])
    return ou


def predict(x_true, coef_params):
    return np.matmul( x_true, coef_params )

def errors(x_true, y_true, coef_params, intercept_params, boundary):
    return mean_squared_error(y_true, predict(x_true, coef_params, intercept_params, boundary))

def grad(x_true, y_true, k_params, shift, eta, boundary, output_k, output_coef, output_kor, output_pred, kp ):
    
    k_diff = np.zeros((2,))
  
    for i in range(2):
            k_params_1 = k_params.copy()
            k_params_2 = k_params.copy()
            k_params_1[i] += shift
            k_params_2[i] -= shift
            output_1 = output(k_params_1)
            output_2 = output(k_params_2)
            if kp > 10:
                drc = 1
            else:
                drc = -1
            k_diff[i] -= drc*(2*np.matmul(np.transpose(np.matmul(np.matmul(x_true,inv(output_kor)),output_coef)), (y_true-output_pred))+np.matmul(np.transpose(output_coef),output_coef)-2*output_k*np.matmul(np.matmul(np.transpose(output_coef),inv(output_kor)),output_coef))*(output_1-output_2)/(2*np.sin(shift))
                        
            #for x, y in zip(x_true, y_true):
            #    coef_diff[j,i] -= (x*(y-predict(x, output_coef, output_intercept, boundary))*(output_1-output_2)/(2*np.sin(shift)))[0]
                         
    k_diff = k_diff*boundary*2/len(y_true)
            
    for i in range(len(k_params)):
        k_params[i] = k_params[i] - eta*k_diff[i]
        
    return k_params           


def OLS(xTrain, yTrain, xTest, yTest):
    # Create linear regression object
    regr = linear_model.LinearRegression(fit_intercept=False)

    # Train the model using the training sets
    regr.fit(xTrain, yTrain)

    yt_pred = regr.predict(xTrain)

    # Make predictions using the testing set
    y_pred = regr.predict(xTest)

    mseTrain = mean_squared_error(yTrain, yt_pred)
    mseTest = mean_squared_error(yTest, y_pred)
    kor = np.matmul(np.transpose(xTrain), xTrain)
    VIF = vif(inv(kor))
    KAPPA = kappa(kor)
    return 0, regr.coef_, regr.intercept_, mseTrain, mseTest, VIF, KAPPA


def khk(beta, sigSqrHat):
    return (sigSqrHat/np.matmul( beta, np.transpose(beta)))    

def khkb(p, beta, sigSqrHat):
    return (p*sigSqrHat/np.matmul( beta, np.transpose(beta)))    
    
def klw(p, beta, Lambda, sigSqrHat):
    return (p*sigSqrHat/np.matmul( Lambda*beta, np.transpose(beta)))    

def ridge_hk(beta, sigSqrHat, xTrain, yTrain, xTest, yTest):
    nr, nc = xTrain.shape
    k = khk(beta, sigSqrHat)
    kor = np.matmul(np.transpose(xTrain), xTrain) + k*np.eye(nc)
    coef = np.matmul(np.matmul(inv(kor),np.transpose(xTrain)),yTrain)
    ytrain_pred_qvm = predict(xTrain, coef)
    ytest_pred_qvm = predict(xTest, coef)

    mseTrain=mean_squared_error(yTrain, ytrain_pred_qvm)
    mseTest=mean_squared_error(yTest, ytest_pred_qvm)
    VIF=vif(inv(kor))
    KAPPA=kappa(kor)
    return k, coef, mseTrain, mseTest, VIF, KAPPA

def ridge_hkb(beta, sigSqrHat, xTrain, yTrain, xTest, yTest):
    nr, nc = xTrain.shape
    k = khkb(nc, beta, sigSqrHat)
    kor = np.matmul(np.transpose(xTrain), xTrain) + k*np.eye(nc)
    coef = np.matmul(np.matmul(inv(kor),np.transpose(xTrain)),yTrain)
    ytrain_pred_qvm = predict(xTrain, coef)
    ytest_pred_qvm = predict(xTest, coef)

    mseTrain=mean_squared_error(yTrain, ytrain_pred_qvm)
    mseTest=mean_squared_error(yTest, ytest_pred_qvm)
    VIF=vif(inv(kor))
    KAPPA=kappa(kor)
    return k, coef, mseTrain, mseTest, VIF, KAPPA

def ridge_lw(beta, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest):
    nr, nc = xTrain.shape
    k = klw(nc, beta, Lambda, sigSqrHat)
    kor = np.matmul(np.transpose(xTrain), xTrain) + k*np.eye(nc)
    coef = np.matmul(np.matmul(inv(kor),np.transpose(xTrain)),yTrain)
    ytrain_pred_qvm = predict(xTrain, coef)
    ytest_pred_qvm = predict(xTest, coef)

    mseTrain=mean_squared_error(yTrain, ytrain_pred_qvm)
    mseTest=mean_squared_error(yTest, ytest_pred_qvm)
    VIF=vif(inv(kor))
    KAPPA=kappa(kor)
    return k, coef, mseTrain, mseTest, VIF, KAPPA

def QRidge(xTrain, yTrain, xTest, yTest, rSeed=123, boun=2, verbose=True):
    np.random.seed(rSeed)
    nr, nc = xTrain.shape
    steps = 10000
    eta1 = 0.00001

    k_params = np.random.uniform(0,1,size=(2,))
    output_k = output(k_params)*boun
    kor = np.matmul(np.transpose(xTrain), xTrain) + output_k*np.eye(nc)
    coef = np.matmul(np.matmul(kor,np.transpose(xTrain)), yTrain)
    KAPPA=kappa(kor)

    ytrain_pred_qvm = predict(xTrain, coef)
    ytest_pred_qvm = predict(xTest, coef)

    rep = 0
    for i in range(steps):
        k_params = grad(xTrain, yTrain, k_params, np.pi/20, eta=eta1, boundary=boun, output_k=output_k, output_coef=coef, output_kor=kor, output_pred = ytrain_pred_qvm, kp=KAPPA)
        old_k = output_k
        output_k = output(k_params)*boun
        kor = np.matmul(np.transpose(xTrain), xTrain) + output_k*np.eye(nc)
        coef = np.matmul(np.matmul(inv(kor),np.transpose(xTrain)),yTrain)
        ytrain_pred_qvm = predict(xTrain, coef)
        ytest_pred_qvm = predict(xTest, coef)

        mseTrain=mean_squared_error(yTrain, ytrain_pred_qvm)
        mseTest=mean_squared_error(yTest, ytest_pred_qvm)
        VIF=vif(inv(kor))
        #old_kappa = KAPPA
        KAPPA=kappa(kor)
        if verbose and (i % 50 == 0):
            print("i= %4d  , k=%.5f   , MSE train= %.5f  , MSE test= %.5f, kappa= %.5f   , VIF= %.5f" % (i , output_k, mseTrain , mseTest, KAPPA, VIF))
        if old_k == output_k: 
            rep = rep+1
        else:
            rep = 0
        if rep > 5:
            break
        # if (i>0) and ((10-old_kappa)*(10-KAPPA)<=0):
        #     break

    return output_k, coef, mseTrain, mseTest, VIF, KAPPA

#simParam_n = np.array( [100, 500, 1000] )
#simParam_rho = np.array( [0.7, 0.8, 0.9, 0.95 ])
#simParam_p = np.array( [(2,1), (3,2), (3,4)])
#simParam_sigma = np.array( [5, 1, 0.5, 0.1])

simParam_n = np.array( [100, 500, 5000] )
simParam_rho = np.array( [0.7, 0.8, 0.9, 0.95, 0.99])
simParam_p = np.array( [(3,2)])
simParam_sigma = np.array( [2, 1, 0.5, 0.2])
simBoun = np.array( [1,2,3,20,50])

def sim(cs):
   ncase = len(simParam_n)*len(simParam_rho)* len(simParam_p)*len(simParam_sigma)
   simR = np.zeros((cs , 11 ))
   simCases = np.zeros((ncase , 9 ))
   j=0
   for n in simParam_n:
       for p in simParam_p:
           for ind,rho in enumerate(simParam_rho):
               for sigma in simParam_sigma:
                   p1 = p[0]
                   p2 = p[1]
                   #beta = [1] * (p1+p2)
                   beta = [0.1,0.5,-0.3,-0.1,0.8]
                   for i in range(cs):
                       x,y = artificial_data(beta, rho, sigma, n, p1, p2)
                    
                       nt = math.ceil(n*0.8)
                       xTrain = x[:nt,:]
                       yTrain = y[:nt]
                       xTest = x[nt:,:]
                       yTest = y[nt:]

                       kor = np.matmul(np.transpose(xTrain), xTrain)
                       Lambda,_ = eig(inv(kor))
                       e = yTrain - np.matmul(xTrain * np.matmul(Lambda, np.transpose(Lambda)),beta)
                       sigSqrHat = np.matmul(e, np.transpose(e))/(nt-(p1+p2))

                       kOLS,coef,intercept,mseTrain,mseTest,vif,kappa = OLS(xTrain, yTrain, xTest, yTest)
                       diffBeta = coef - beta    
                       mseBetaOLS = np.matmul( diffBeta, np.transpose(diffBeta))                         

                       kHK,coef,mseTrain,mseTest,vif,kappa = ridge_hk(beta, sigSqrHat, xTrain, yTrain, xTest, yTest)
                       diffBeta = coef - beta    
                       mseBetaKHK = np.matmul( diffBeta, np.transpose(diffBeta))                         

                       kHKB,coef,mseTrain,mseTest,vif,kappa = ridge_hkb(beta, sigSqrHat, xTrain, yTrain, xTest, yTest)
                       diffBeta = coef - beta    
                       mseBetaKHKB = np.matmul( diffBeta, np.transpose(diffBeta))                         

                       kLW,coef,mseTrain,mseTest,vif,kappa = ridge_lw(beta, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest)
                       diffBeta = coef - beta    
                       mseBetaKLW = np.matmul( diffBeta, np.transpose(diffBeta))                         

                       kQR,coef,mseTrain,mseTest,vif,kappa = QRidge(xTrain, yTrain, xTest, yTest, rSeed=123+i, boun=simBoun[ind], verbose=False)
                       diffBeta = coef - beta    
                       mseBeta = np.matmul( diffBeta, np.transpose(diffBeta))                         
                       simR[i] = np.array([i,n,rho,sigma,kQR,mseBetaOLS, mseBetaKHK,mseBetaKHKB,mseBetaKLW, mseBeta,kappa])
                       print("%4d, n= %4d, rho= %.2f, sigma= %.1f, kHK= %.5f, kQR= %.5f, mseBetaOLS= %.5f, mseBeta= %.5f" % (i, n, rho, sigma, kHK, kQR, mseBetaOLS, mseBeta))
                   simCases[j] = np.array([n,rho,sigma,kQR,np.average(simR[:,5]),np.average(simR[:,6]),np.average(simR[:,7]),np.average(simR[:,8]),np.average(simR[:,9])])
                   print(">>>>>  OLS= %.5f,  HK= %.5f,  HKB= %.5f,  LW= %.5f,  QRidge= %.5f" % (simCases[j,4], simCases[j,5], simCases[j,6], simCases[j,7], simCases[j,8]))
                   j=j+1

   header = ['n', 'rho', 'sigma', 'kQRidge', 'OLS', 'HK', 'HKB', 'LW', 'QRidge' ]
   with open('simulation.csv', 'w', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(simCases)
   return simCases

def diabetes():
   X, y = datasets.load_diabetes(return_X_y=True)
   y = (y - np.min(y)) / (np.max(y) - np.min(y))
# Use only one feature
#X = X[:, np.newaxis, 2]

   nr, nc = X.shape
   for i in range(nc):
       X[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))

   # Split the data into training/testing sets
   xTrain = X[:354,2:]    # Use only feature 2 and 3
   xTest = X[354:,2:]
   nr, nc = xTrain.shape

   # Split the targets into training/testing sets
   yTrain = y[:354]
   yTest = y[354:]    

   print("Method", "k", "beta","intercept","mse Train","mes Test", "VIF", "kappa")
   kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS = OLS(xTrain, yTrain, xTest, yTest)
   print("OLS    ",kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS)
   
   kor = np.matmul(np.transpose(xTrain), xTrain)
   Lambda,_ = eig(inv(kor))
   e = yTrain - np.matmul(xTrain * np.matmul(Lambda, np.transpose(Lambda)),coefOLS)
   sigSqrHat = np.matmul(e, np.transpose(e))/(nr-nc)
   
   kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK = ridge_hk(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HK     ",kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK)

   kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB = ridge_hkb(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HKB    ",kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB)

   kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW = ridge_lw(coefOLS, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("LW     ",kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW )

   kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR = QRidge(xTrain, yTrain, xTest, yTest, rSeed=123, boun=20, verbose=False)
   print("QRidge ",kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR) 
  
   return 0                      


def watermelon():

   wm = np.loadtxt(open("watermelon.csv", "rb"), delimiter=",", skiprows=1)
   X = wm[:,1:]
   y = wm[:,0]

   nr, nc = X.shape
   for i in range(nc):
       X[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))
   y = (y - np.min(y)) / (np.max(y) - np.min(y))

   # Split the data into training/testing sets
   xTrain = X[:18,:]    # Use only feature 2 and 3
   xTest = X[18:,:]
   nr, nc = xTrain.shape

   # Split the targets into training/testing sets
   yTrain = y[:18]
   yTest = y[18:]    

   print("Method", "k", "beta","intercept","mse Train","mes Test", "VIF", "kappa")
   kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS = OLS(xTrain, yTrain, xTest, yTest)
   print("OLS    ",kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS)
   
   kor = np.matmul(np.transpose(xTrain), xTrain)/nr
   Lambda,_ = eig(kor)
   e = yTrain - np.matmul(xTrain * np.matmul(Lambda, np.transpose(Lambda)),coefOLS)
   sigSqrHat = np.matmul(e, np.transpose(e))/(nr-nc)
   
   kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK = ridge_hk(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HK     ",kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK)

   kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB = ridge_hkb(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HKB    ",kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB)

   kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW = ridge_lw(coefOLS, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("LW     ",kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW )

   kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR = QRidge(xTrain, yTrain, xTest, yTest, rSeed=123, boun=1, verbose=False)
   print("QRidge ",kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR) 
  
   return 0      


def gp():

   gpd = np.loadtxt(open("gp.csv", "rb"), delimiter=";")
   X = gpd[:,1:]
   y = gpd[:,0]

   nr, nc = X.shape
#   for i in range(nc):
#       X[:,i] = (X[:,i] - np.min(X[:,i])) / (np.max(X[:,i]) - np.min(X[:,i]))
   y = (y - np.min(y)) / (np.max(y) - np.min(y))

   # Split the data into training/testing sets
   xTrain = X[:350,:]    # Use only feature 2 and 3
   xTest = X[350:,:]
   nr, nc = xTrain.shape

   # Split the targets into training/testing sets
   yTrain = y[:350]
   yTest = y[350:]    

   print("Method", "k", "beta","intercept","mse Train","mes Test", "VIF", "kappa")
   kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS = OLS(xTrain, yTrain, xTest, yTest)
   print("OLS    ",kOLS,coefOLS,interceptOLS,mseTrainOLS,mseTestOLS,vifOLS,kappaOLS)
   
   kor = np.matmul(np.transpose(xTrain), xTrain)
   Lambda,_ = eig(inv(kor))
   e = yTrain - np.matmul(xTrain * np.matmul(Lambda, np.transpose(Lambda)),coefOLS)
   sigSqrHat = np.matmul(e, np.transpose(e))/(nr-nc)
   
   kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK = ridge_hk(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HK     ",kHK,coefHK,mseTrainHK,mseTestHK,vifHK,kappaHK)

   kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB = ridge_hkb(coefOLS, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("HKB    ",kHKB,coefHKB,mseTrainHKB,mseTestHKB,vifHKB,kappaHKB)

   kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW = ridge_lw(coefOLS, Lambda, sigSqrHat, xTrain, yTrain, xTest, yTest)
   print("LW     ",kLW,coefLW,mseTrainLW,mseTestLW,vifLW,kappaLW )

   kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR = QRidge(xTrain, yTrain, xTest, yTest, rSeed=123, boun=50, verbose=False)
   print("QRidge ",kQR,coefQR,mseTrainQR,mseTestQR,vifQR,kappaQR) 
  
   return 0   

       

                    
