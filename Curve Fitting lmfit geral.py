import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate  
import pandas as pd
from lmfit import minimize, Parameters, Parameter, report_fit

## inserção
def f(t, y, paras):
    """
    Your system of differential equations
    """

    S = y[0] #kgDQO/m^3
    X = y[1]
    P = y[2]

    try:
        S_in = paras['S_in'].value #kgDQO/m^3
        mumax_X = paras['mumax_X'].value #dia^-1
        K_S = paras['K_S'].value #kgDQO/m^3
        Y_X_S = paras['Y_X_S'].value #kg_DQO_X/kg_DQO_S
        Y_P_S = paras['Y_P_S'].value #kg_DQO_P/kg_DQO_S
        Y_S_X = paras['Y_S_X'].value #kgDQO_S/kg_DQO_X
        k_dec = paras['k_dec'].value #dia^-1
        D = paras['D'].value #dia^-1
        a = paras['a'].value #%
        b = paras['b'].value #yield



    except KeyError:
        S_in, mumax_X, K_S, Y_X_S, Y_P_S, k_dec, D, a = paras
    # the model equations
    
    mu = (mumax_X*S)/((K_S/(1-a+1e-16)) + S + 1e-16) #dia^-1
    dS_dt = (D)*(S_in - S) + (-(mu/(Y_X_S+1e-16)) + Y_S_X*k_dec)*X
    dX_dt = (D)*(-X) + ((Y_X_S*mu)-k_dec)*X
    dP_dt = (D)*(-P) + b*Y_P_S*(mu*X)/(Y_X_S+1e-16)

    return [dS_dt, dX_dt, dP_dt]


def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = scipy.integrate.solve_ivp(f, t, x0, 'RK45', args=(paras), t_eval=np.linspace(min(t), max(t), 25))
    print(x)
    return x


def residual(paras, t, data, x0):

    """
    compute the residual between actual data and fitted data
    """

    # x0 = paras['x10'].value, paras['x20'].value, paras['x30'].value
    model = g(t, x0, [paras])
    model = pd.DataFrame(model.y).transpose()
    print('parametros: \n {}'.format(paras))
    print('resultado da integração: \n {}'.format(model))

    # you only have data for one of your variables
    S_model = model.iloc[:, 0]
    P_model = model.iloc[:, 2] 
    return ((P_model - data.iloc[:, 1])+(S_model-data.iloc[:,0])).ravel()


# initial conditions
x10 = 42500.
x20 = 25200.
x30 = 0.
y0 = [x10, x20, x30]

# measured data
t_measured = np.linspace(0, 240, 25)
dados_P = pd.read_excel(r"C:\Users\claro\OneDrive\Documentos\Modelos Personalizados do Office\Exp Rao et al .xlsx", header=None, names=['tempo', 'concentração'], decimal=',')
dados_S = pd.read_excel(r"C:\Users\claro\OneDrive\Documentos\UTFPR\TCC\Código\Rao et al. substrato.xlsx", header=None, names=['tempo', 'concentração'], decimal=',')

data = [dados_S.iloc[:,1],dados_P.iloc[:,1]]
data = pd.DataFrame(data).transpose()
print(data)
plt.figure().suptitle('RK45 + trust-constr')
plt.scatter(t_measured, dados_P['concentração'], marker='o', color='b', label='P exp', s=75)
plt.scatter(t_measured, dados_S['concentração'], marker='o', color='y', label='S exp', s=75)
# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()

params.add('S_in', value=1e-16, vary=False)
params.add('mumax_X', value=0.3, min=1e-16, max=10)
params.add('K_S', value=0.10, min=1e-16)
params.add('Y_X_S', value=0.7, min=1e-16)
params.add('Y_P_S', value=0.3, min=1e-16)
params.add('Y_S_X', value=0.2, min=1e-16)
params.add('k_dec', value=0.00015, min=1e-16)
params.add('D', value=1e-16, vary=False)
params.add('a', value=0.5, min=1e-16, max=1)
params.add('b', value=0.5, min=1e-16)

print(pd.DataFrame(g([0, 240], y0, [params]).y).transpose())
# fit model
result = minimize(residual, params, args=([0, 240], data, y0), method='trust-krylov')  # leastsq nelder
# check results of the fit
data_fitted = pd.DataFrame(g([0, 240], y0, [result.params]).y).transpose()

# plot fitted data
plt.plot(np.linspace(0., 240., 25), data_fitted.iloc[:, 2], '-', linewidth=2, color='red', label='fitted data')
plt.legend()
plt.xlim([0, max(t_measured)])
plt.ylim([0, 1.1 * max(data_fitted.iloc[:, 0])])
# display fitted statistics
report_fit(result)

plt.show()

