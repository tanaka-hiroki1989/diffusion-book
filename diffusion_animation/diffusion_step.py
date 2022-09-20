import numpy as np

def diffusion_step(points,beta):
    k1 = np.sqrt(1-beta)
    k2 = np.sqrt(beta)
    return k1 * points + k2 * np.random.randn(len(points),2)