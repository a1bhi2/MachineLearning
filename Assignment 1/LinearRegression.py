import numpy as np
import pandas as pd
from matplotlib import pyplot as plt 

data = np.loadtxt("ex1data1.txt",delimiter=',')
p=data[:,0]
X=np.array([np.ones(p.size).transpose(),p])
d=pd.DataFrame(data=X.transpose())

y=pd.DataFrame(data=data[:,1])
theta = np.zeros((2,1))
m=y.size
alpha=0.01

def plotall():
    plt.plot(data[:,0],np.matmul(d,theta))
    plt.scatter(data[:,0],data[:,1])
    
    plt.show()
	
def costfunction(theta,i):
    J=(1/(2*m))*np.sum(np.square(np.matmul(d ,theta)-y))
    plt.scatter(i,J)
    #print(J)
    

def gradientDescent(theta,num_iters):
    J_history = np.zeros((num_iters,1))
    for i in range (num_iters):
        theta = theta - (alpha * 1/m) * np.transpose((np.matmul(((np.matmul(d,theta)-y).transpose()),d)))
        J_history[i] = costfunction(theta,i);
    plt.show()
    return (theta)
      
theta=gradientDescent(theta,1500)
plotall()
