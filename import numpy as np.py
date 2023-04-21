import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
import SALib
from SALib.sample import saltelli
from SALib.analyze import sobol
from mpmath import mp

def model_SA (t, y, S_in, mumax_X, K_S, Y_X_S, Y_P_S, k_dec, D):
    ##são considerados os balanços de massa de três componentes:
    S = y[0] #g_DQO_S/m^3
    X = y[1] #g_DQO_X/m^3
    P = y[2] #g_DQO_P/m^3

    ##considerada 
    mu = mp.mpf(mumax_X)*mp.mpf(S)/mp.mpf((K_S + S)) #dia^-1
        

    dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
    dX_dt = (D)*(-X) + (mu-k_dec)*X
    dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X)


    return [mp.mpf(dS_dt), mp.mpf(dX_dt), mp.mpf(dP_dt)]

def g_SA (t, X0, S_in, mumax_X, K_S, Y_X_S, Y_P_S, k_dec, D):

    model_result = scipy.integrate.solve_ivp(model_SA, t, X0, method='LSODA', args=(S_in, mumax_X, K_S, Y_X_S, Y_P_S, k_dec, D))
    S_model, X_model, P_model = model_result.y
    S_model = np.longdouble(S_model)
    X_model = np.longdouble(X_model)
    P_model = np.longdouble(P_model)
    model_result = [S_model, X_model, P_model]
    model_result = pd.DataFrame(model_result).transpose()
    
    return model_result

x = g_SA([0, 240], [42500., 25500., 0.], 0.37646484375, 5.2501708984375, 29.2236328125, 0.13935546875, 0.54638671875, 0.415771484375, 0.0096337890625)
print(x)
print(x.iloc[-1])
log = np.log10(x[:][0])
print(log)
plt.plot(x.index, np.log10(x[:][2]))
plt.show()