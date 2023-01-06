#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
author: Y. F. Zhang
"""

import numpy as np
# import matplotlib.pyplot as plt
from timeit import default_timer as timer
from numba import cuda, vectorize, jit
from numpy import sin, cos, tan ,cosh, tanh, sinh, abs, exp, mean, pi, prod, sqrt, sum



class GWO:
    def __init__(self):
        self.wolf_num = 1000
        self.max_iter = 1000
        self.dim = 30
        self.lb = -30*np.ones((self.dim,))
        self.ub = 30*np.ones((self.dim,))
        self.alpha_pos = np.zeros((1,self.dim))
        self.beta_pos = np.zeros((1, self.dim))
        self.delta_pos = np.zeros((1, self.dim))
        self.alpha_score = np.inf
        self.beta_score = np.inf
        self.delta_score = np.inf
        self.convergence_curve = np.zeros((self.max_iter,))
        self.position = np.zeros((self.wolf_num,self.dim))
      
    #@jit(target_backend='cuda')
    def run(self):
        start = timer()
        count = 0
        step = 0
        self.init_pos()
        while count < self.max_iter:
            for i in range(self.wolf_num):
                flag_ub = self.position[i,:] > self.ub
                flag_lb = self.position[i,:] < self.lb
                self.position[i,:] = self.position[i,:]*(~(flag_lb+flag_ub))+flag_ub*self.ub+flag_lb*self.lb
                fitness = self.griewank(self.position[i,:])
                if fitness < self.alpha_score:
                    self.alpha_score = fitness
                    self.alpha_pos = self.position[i,:]
                elif fitness < self.beta_score:
                    self.beta_score = fitness
                    self.beta_pos = self.position[i,:]
                elif fitness < self.delta_score:
                    self.delta_score = fitness
                    self.delta_pos = self.position[i,:]
            a = 2 - count*(2/self.max_iter)
            for i in range(self.wolf_num):
                for j in range(self.dim):
                    alpha = self.update_pos(self.alpha_pos[j],self.position[i,j],a)
                    beta = self.update_pos(self.beta_pos[j], self.position[i, j], a)
                    delta = self.update_pos(self.delta_pos[j], self.position[i, j], a)
                    self.position[i, j] = sum(np.array([alpha, beta, delta]) * np.array([1/3,1/3,1/3]))
                    print("###############\n" + "Alpha: " + str(alpha) + "\nBeta: " + str(beta) + "\nDelta: " + str(delta))
            step += 1
            count += 1
            self.convergence_curve[count-1] = self.alpha_score
        # self.plot_results()
        print("On a GPU: ", timer() - start)

    # @jit(target_backend='cuda')
    def init_pos(self):
        for i in range(self.wolf_num):
            for j in range(self.dim):
                self.position[i,j] = np.random.rand()*(self.ub[j]-self.lb[j])+self.lb[j]

    @staticmethod
    #@jit(target_backend='cuda')
    def update_pos(v1,v2,a):
        A = 2*np.random.rand()*a-a
        C = 2*np.random.rand()
        temp = np.abs(C*v1-v2)
        return v1 - A*temp

    #@jit(target_backend='cuda')
    def plot_results(self):
        plt.style.use('seaborn-v0_8-darkgrid')
        plt.plot(range(1,self.max_iter+1),self.convergence_curve,'g.--')
        plt.xlabel('iteration')
        plt.ylabel('fitness')
        plt.title('GWO fitness curve')
        plt.show()

    @staticmethod
    #@jit(target_backend='cuda')
    def rosenbrock(x):
        dim, s = 30, 0
        for i in range(len(x)-1):
            s += 100*(x[i+1]-x[i]**2)**2+(x[i]-1)**2
        return s
    
    @staticmethod
    #@jit(target_backend='cuda')
    def ackley( x, a=20, b=0.2, c=2*pi ):
        x = np.asarray_chkfinite(x)  # ValueError if any NaN or Inf
        n = len(x)
        s1 = sum( x**2 )
        s2 = sum( cos( c * x ))
        return -a*exp( -b*sqrt( s1 / n )) - exp( s2 / n ) + a + exp(1)
    
    @staticmethod
    #@jit(target_backend='cuda')
    def rastrigin( x ):  # rast.m
        x = np.asarray_chkfinite(x)
        n = len(x)
        return 10*n + sum( x**2 - 10 * cos( 2 * pi * x ))
    
    @staticmethod
    #@jit(target_backend='cuda')
    def griewank( x, fr=4000 ):
        x = np.asarray_chkfinite(x)
        n = len(x)
        j = np.arange( 1., n+1 )
        s = sum( x**2 )
        p = prod( cos( x / sqrt(j) ))
        return s/fr - p + 1

if __name__ == "__main__":
    gwo = GWO()
    gwo.run()

