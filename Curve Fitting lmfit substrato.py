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

    S = y[0]
    X = y[1]
    P = y[2]

    try:
        S_in = paras['S_in'].value
        mumax_X = paras['mumax_X'].value
        K_S = paras['K_S'].value
        Y_X_S = paras['Y_X_S'].value #g_DQO_X/g_DQO_S
        Y_P_S = paras['Y_P_S'].value #g_DQO_P/g_DQO_S
        k_dec = paras['k_dec'].value #dia^-1
        D = paras['D'].value #dia^-1



    except KeyError:
        S_in, mumax_X, K_S, Y_X_S, Y_P_S, k_dec, D = paras
    # the model equations

    mu = mumax_X*S/(K_S + S) #dia^-1
    dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
    dX_dt = (D)*(-X) + (mu-k_dec)*X
    dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X)
    return [dS_dt, dX_dt, dP_dt]


def g(t, x0, paras):
    """
    Solution to the ODE x'(t) = f(t,x,k) with initial condition x(0) = x0
    """
    x = scipy.integrate.solve_ivp(f, t, x0, args=(paras), t_eval=np.linspace(min(t), max(t), 25))
    return x


def residual(paras, t, data, x0):

    """
    compute the residual between actual data and fitted data
    """

    # x0 = paras['x10'].value, paras['x20'].value, paras['x30'].value
    model = g(t, x0, [paras])
    model = pd.DataFrame(model.y).transpose()

    # you only have data for one of your variables
    x2_model = model.iloc[:, 0]
    return (x2_model - data).ravel()


# initial conditions
x10 = 42500.
x20 = 25200.
x30 = 0.
y0 = [x10, x20, x30]

# measured data
t_measured = np.linspace(0, 240, 25)
dados_P = pd.read_excel(r"C:\Users\claro\OneDrive\Documentos\Modelos Personalizados do Office\Exp Rao et al .xlsx", header=None, names=['tempo', 'concentração'], decimal=',')
dados_S = pd.read_excel(r"C:\Users\claro\OneDrive\Documentos\UTFPR\TCC\Código\Rao et al. substrato.xlsx", header=None, names=['tempo', 'concentração'], decimal=',')
plt.figure()
plt.scatter(t_measured, dados_S['concentração'], marker='o', color='b', label='measured data', s=75)

# set parameters including bounds; you can also fix parameters (use vary=False)
params = Parameters()

params.add('S_in', value=100., min=0.0001)
params.add('mumax_X', value=0.3, min=0.0001)
params.add('K_S', value=0.10, min=0.0001)
params.add('Y_X_S', value=0.7, min=0)
params.add('Y_P_S', value=0.3, min=0)
params.add('k_dec', value=0.00015, min=0)
params.add('D', value=0., vary=False)

print(pd.DataFrame(g([0, 240], y0, [params]).y).transpose())
# fit model
result = minimize(residual, params, args=([0, 240], dados_S['concentração'], y0), method='leastsq')  # leastsq nelder
# check results of the fit
data_fitted = pd.DataFrame(g([0, 240], y0, [result.params]).y).transpose()

# plot fitted data
plt.plot(np.linspace(0., 240., 25), data_fitted.iloc[:, 0], '-', linewidth=2, color='red', label='fitted data')
plt.legend()
plt.xlim([0, max(t_measured)])
plt.ylim([0, 1.1 * max(data_fitted.iloc[:, 0])])
# display fitted statistics
report_fit(result)

plt.show()
