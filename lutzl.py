import numpy as np
import matplotlib.pyplot as plt



def RK4integrate(f,t,y0):
    y = np.asarray(len(t)*[y0]);
    for i in range(len(t)-1):
        h = t[i+1]-t[i];
        k1=h*f(t[i],y[i]);
        k2=h*f(t[i]+0.5*h,y[i]+0.5*k1);
        k3=h*f(t[i]+0.5*h,y[i]+0.5*k2);
        k4=h*f(t[i+1],y[i]+k3);
        y[i+1,:]=y[i]+(k1+2*k2+2*k3+k4)/6;
    return y

def p(t): return np.exp(-t**2/2)
def odefunc(t,x): return -t*x


fig, ax = plt.subplots(2,1,figsize=(12,10))
t0, tmax=-10, 10
for h in [0.1, 0.05, 0.025, 0.01, 0.005 ][::-1]:
    t = np.arange(t0,tmax,h);
    y = RK4integrate(odefunc, t, np.array([p(t[0])]));
    ax[0].plot(t,y[:,0],'-o', ms=1+13*h, label="h=%.3g"%h);
    ax[1].plot(t,(y[:,0]/p(t)-1)/h**4,'-o', ms=1+16*h, label="h=%.3g"%h);
for gr in ax: gr.grid(); gr.legend();
plt.show();