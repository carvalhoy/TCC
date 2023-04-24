# importanto bibliotecas:
import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from lmfit import minimize, Parameters, Parameter, report_fit



def model (t, x, params):
    # S_in, mu_max_X, K_S, Y_X_S, Y_P_S, k_dec, V, q = parameters_model

    S = x[0] #g_DQO_S/m^3
    X = x[1] #g_DQO_X/m^3
    P = x[2] #g_DQO_P/m^3

    try:
        #parâmetros do modelo
        S_in = params['S_in'].value
        mu_max_X = params['mumax_X'].value #dia^-1
        K_S = params['K_S'].value #g_DQO_S/m^3
        Y_X_S = params['Y_X_S'].value #g_DQO_X/g_DQO_S
        Y_P_S = params['Y_P_S'].value #g_DQO_P/g_DQO_S
        k_dec = params['k_dec'].value #dia^-1
        D = params['D'].value #dia^-1
    
    except KeyError:
        S_in, mu_max_X, K_S, Y_X_S, Y_P_S, k_dec, D = params
    
    ##definindo a reação
    mu = mu_max_X*S/(K_S + S) #dia^-1
     
    dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
    dX_dt = (D)*(-X) + (mu-k_dec)*X
    dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X)

    return [dS_dt, dX_dt, dP_dt]

##abrindo a planilha do excel com colunas tempo e concentração
def ajustarXlsx(caminho, parametrosDadosXlsx):
    [tempoInicial, tempoFinal, totalPontos, digitosDecimais] = parametrosDadosXlsx
    dados = pd.read_excel(caminho, header=None, names=['tempo', 'concentração'], decimal=',')
    dados['tempo'] = np.linspace(tempoInicial, tempoFinal, totalPontos)
    dados['concentração'] = round(dados['concentração'], digitosDecimais)
    return dados

# Integração usando paras e converte o resultado de matriz numpy em transforma em dataframe para obter a coluna de índice 2 - produtos:
def residual(paras, t_step, data, max_step, t_solve_ivp, x, indice, metodoIntegracao):
    model = integracao(metodoIntegracao, t_step, paras, max_step, t_solve_ivp, x)
    model = pd.DataFrame(model.y).transpose()
    modelVariaveldoModelo = model.iloc[:, indice]
    error = (modelVariaveldoModelo - data).ravel()
    ##print(error)
    return error
    
def integracao(metodoIntegracao, t_step, params, max_step, t_solve_ivp, x):
    solve_ivp = scipy.integrate.solve_ivp(model, t_step, x, metodoIntegracao, t_eval=t_solve_ivp, args=[params], max_step=max_step)
    return solve_ivp
    
def plotagem(solve_ivp, dados_P, dados_S):
    #abrindo a tupla retornada em três variáveis
    S_solve_ivp, X_solve_ivp, P_solve_ivp = solve_ivp.y
    #plotando 
    plt.plot(solve_ivp.t, S_solve_ivp, 'o', color='red')
    plt.plot(dados_P['tempo'], dados_P['concentração'])
    plt.plot(dados_S['tempo'], dados_S['concentração']) 
    plt.plot(solve_ivp.t, X_solve_ivp, 'o', color='green')
    plt.plot(solve_ivp.t, P_solve_ivp, 'o', color='blue')
    plt.show()
        
def main():
    parametrosDadosXlsx = [0, 240, 25, 3] #tempo inicial, tempo final, número de pontos, algarismos significativos
    dados_P = ajustarXlsx("../xlsx1/produto.xlsx", parametrosDadosXlsx)
    dados_S = ajustarXlsx("../xlsx1/substrato.xlsx", parametrosDadosXlsx)

    #condições iniciais das variáveis dos balanços, intervalo de tempo considerado, 
    x = [42500.0, 25200.0, 0.0]
    t_step = [0, 240]
    t_solve_ivp = dados_P['tempo']
    #intervalo de integração, max_step e dados no tempo a serem retornados pela função integrate.solve_ivp:
    metodoIntegracao = 'RK45'
    max_step = 0.05
    
    ##definindo os parâmetros declarados como argumento para o solve_ivp
    ##params = (1.46484375e-03, 6.11696777e+00, 2.82084961e+02,   5.57128906e-01, 4.22119141e-01, 2.41699219e-03, 0.)

    ## Definindo parâmetros para Curve Fitting:
    paras = Parameters()
    paras.add('S_in', value=0., vary=False)
    paras.add('mumax_X', value=0.3, min=0.0001)
    paras.add('K_S', value=0.10, min=0.0001)
    paras.add('Y_X_S', value=0.7, min=0)
    paras.add('Y_P_S', value=0.3, min=0)
    paras.add('k_dec', value=0.00015, min=0)
    paras.add('D', value=0., vary=False)
    
    metodoMinimizacao = 'leastsq'
    
    ## Otimização para Substrato (0), Biomassa (1), Produto (2):
    indice = [0, 1, 2]
    dadoOtimizacao = [dados_S['concentração'], dados_P['concentração']]
    
    resultProduto = minimize(residual, paras, args=(t_step, dadoOtimizacao[1], max_step, t_solve_ivp, x, indice[2], metodoIntegracao), method=metodoMinimizacao)  # leastsq
    report_fit(resultProduto) 
    
    
    #execução da função de integração e plotagem do gráfico: 
    ##plotagem(integracao(metodoIntegracao, t_step, params, max_step, t_solve_ivp, x), dados_P, dados_S)
    
main()