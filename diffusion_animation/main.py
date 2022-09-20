import numpy as np
from diffusion_step import diffusion_step
from plot import plot
from tqdm import tqdm 

#data = np.random.rand(1000,2) + [2.0,2.0]
data1 = np.random.rand(1000,2) + [2.0,2.0]
data2 = np.random.rand(1000,2) + [3.0,3.0]
#c = np.array([0]*1000)
c1 = np.array([0]*1000)
c2 = np.array([1]*1000)

data=np.concatenate([data1,data2])
c=np.concatenate([c1,c2])

M=4000
h = data
beta =np.linspace(0.001,0.005,M)
for m in tqdm(range(M)):
    h = diffusion_step(h,beta[m])
    plot(h,m,c)