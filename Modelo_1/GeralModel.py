# importanto bibliotecas:
import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt


def model (t, x, params):
    # S_in, mu_max_X, K_S, Y_X_S, Y_P_S, k_dec, V, q = parameters_model

    S = x[0] #g_DQO_S/m^3
    X = x[1] #g_DQO_X/m^3
    P = x[2] #g_DQO_P/m^3

    #parâmetros do modelo
    S_in = params[0]
    mu_max_X = params[1] #dia^-1
    K_S = params[2] #g_DQO_S/m^3
    Y_X_S = params[3] #g_DQO_X/g_DQO_S
    Y_P_S = params[4] #g_DQO_P/g_DQO_S
    k_dec = params[5] #dia^-1
    D = params[6] #dia^-1
    
    ##definindo a reação
    mu = mu_max_X*S/(K_S + S) #dia^-1
     
    dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
    dX_dt = (D)*(-X) + (mu-k_dec)*X
    dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X)

    return dS_dt, dX_dt, dP_dt

##abrindo a planilha do excel com colunas tempo e concentração
def ajustarXlsx(caminho, parametrosDadosXlsx):
    [tempoInicial, tempoFinal, totalPontos, digitosDecimais] = parametrosDadosXlsx
    dados = pd.read_excel(caminho, header=None, names=['tempo', 'concentração'], decimal=',')
    dados['tempo'] = np.linspace(tempoInicial, tempoFinal, totalPontos)
    dados['concentração'] = round(dados['concentração'], digitosDecimais)
    return dados

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
    dados_P = ajustarXlsx("./xlsx1/produto.xlsx", parametrosDadosXlsx)
    dados_S = ajustarXlsx("./xlsx1/substrato.xlsx", parametrosDadosXlsx)

    ##definindo os parâmetros declarados como argumento para o solve_ivp
    params = (1.46484375e-03, 6.11696777e+00, 2.82084961e+02,   5.57128906e-01, 4.22119141e-01, 2.41699219e-03, 0.)

    #condições iniciais das variáveis dos balanços
    x = [42500.0, 25200.0, 0.0]

    #intervalo de integração, max_step e dados no tempo a serem retornados pela função integrate.solve_ivp:
    metodoIntegracao = 'RK45'
    t_step = [0, 240]
    max_step = 0.05
    t_solve_ivp = dados_P['tempo']
    
    #execução da função de integração e plotagem do gráfico: 
    plotagem(integracao(metodoIntegracao, t_step, params, max_step, t_solve_ivp, x), dados_P, dados_S)
    
main()