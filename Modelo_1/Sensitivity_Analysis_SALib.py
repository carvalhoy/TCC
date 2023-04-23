import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
import SALib
from SALib.sample import saltelli, latin
from SALib.analyze import sobol, pawn
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

x = g_SA([0, 240], [42500., 25500., 0], 100, 1e-4, 9593235, 1.1355e-5, 12607.5, 0.00597, 0)
print(x)
print(x.iloc[-1])
plt.plot()
problem = {
    'num_vars': 5,
    'names': ['mumax_X', 'K_S', 'Y_X_S', 'Y_P_S', 'k_dec'],
    'bounds': [[0.01, 16],
               [0.01, 630],
               [0.01, 0.3],
               [0.01, 1],
               [0.01, 0.5]]
}

param_values = saltelli.sample(problem, 512, calc_second_order=True, skip_values=1024)

print(param_values)
print('param_values.shape:')
print(param_values.shape)
Y = np.zeros([len(param_values), 3])

param_values = pd.DataFrame(param_values)

print(param_values.dtypes)
print(Y)
X0 = [42500., 25500., 0]
X0 = np.array(X0)
# break_list = []
# print(g_SA([0, 240], [42500., 25500., 0.], 0.37646484375, 5.2501708984375, 29.2236328125, 0.13935546875, 0.54638671875, 0.415771484375, 0.0096337890625))
# print(X0.shape[0])
# A = g_SA([0, 240], [42500., 25500., 0.], 0.37646484375, 5.2501708984375, 29.2236328125, 0.13935546875, 0.54638671875, 0.415771484375, 0.0096337890625)
# # print(A[~np.isnan(A).any(axis=1),:])
# print(A.last_valid_index())
# print(A.iloc[A.last_valid_index() - 10, :])
# for i in range(len(param_values)):
#     solve = g_SA([0, 240], [42500., 25500., 0], param_values.iloc[i][0], param_values.iloc[i][1], param_values.iloc[i][2], param_values.iloc[i][3], param_values.iloc[i][4], param_values.iloc[i][5], param_values.iloc[i][6])
#     Y[i][:] = solve.iloc[-1]

#     if np.isnan(solve.iloc[-1]).any():
        
#         Y[i][:] = solve.iloc[solve.last_valid_index(), :]/1e+100
#         params_break = [param_values.iloc[i][0], param_values.iloc[i][1], param_values.iloc[i][2], param_values.iloc[i][3], param_values.iloc[i][4], param_values.iloc[i][5], param_values.iloc[i][6]]
#         break_list.append(params_break)

#     print(param_values.iloc[i][:])
#     print(Y[i][:])

# print(Y.np.nanmean())
# print(Y[0, :])
# print (break_list)

#PAWN analysis
sample_latin = latin.sample(problem, 8000)
Y = np.zeros([len(sample_latin), 3])
print(Y.shape)

print(sample_latin)
for u, i in enumerate(sample_latin):
    solve_pawn = g_SA([0, 240], [42500., 25500., 0.], 0., i[0], i[1], i[2], i[3], i[4], 0.)
    print(solve_pawn.iloc[-1])
    Y[u][:] = solve_pawn.iloc[-1]

print(Y)
print('\n PAWN analysis \n')

Si_S_pawn = pawn.analyze(problem, sample_latin, Y[:, 0], print_to_console=True)
Si_S_pawn.plot()
plt.show()

#Sobol analysis
print('\n\n====S_in Sobol output====\n\n')  
Si_S_in = sobol.analyze(problem, Y[:,0], print_to_console=True)

# plt.plot(x.iloc[:, 3], x.iloc[:, 0])
# plt.show()

