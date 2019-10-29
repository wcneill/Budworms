from Applications import vfield as vf
import numpy as np

def allee_pop(t, n, a=2, b=1.5, r=.5):
    return n*(r - a*(n - b)**2)

def log_pop(t, n, r=0.5, k=1):
    return r*n*(1 - n/k)

#Plot equation displaying Allee effect
vf.vf_grapher(allee_pop, 0, 3, .1, [.9, 1.5, 2.2])

# Then compare to logistic equation
vf.vf_grapher(log_pop, 0, 3, .1, 0.5)
