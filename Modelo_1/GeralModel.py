# importanto bibliotecas:
import numpy as np
import scipy.integrate
import scipy.optimize
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime as dt
from lmfit import minimize, Parameters, Parameter, report_fit
from simple_chalk import chalk



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
    dados['concentração'] = round(dados['concentração']/1000, digitosDecimais)
    return dados

# Integração usando paras e converte o resultado de matriz numpy em transforma em dataframe para obter a coluna de índice 2 - produtos:
def residual(paras, t_step, data, t_solve_ivp, x, rtol, atol, indice, metodoIntegracao):
    model = integracao(metodoIntegracao, t_step, paras, t_solve_ivp, x, rtol, atol)
    model = pd.DataFrame(model.y).transpose()
    modelVariaveldoModelo = model.iloc[:, indice]
    error = (modelVariaveldoModelo - data).ravel()
    ##print(error)
    return error
    
def integracao(metodoIntegracao, t_step, params, t_solve_ivp, x, rtol, atol):
    solve_ivp = scipy.integrate.solve_ivp(model, t_step, x, metodoIntegracao, t_eval=t_solve_ivp, args=[params], rtol=rtol, atol=atol)
    return solve_ivp
    
def plotagem(solve_ivp, dados_P, dados_S, metodo_integracao, metodo_otimizacao):
    #abrindo a tupla retornada em três variáveis
    S_solve_ivp, X_solve_ivp, P_solve_ivp = solve_ivp.y
    #plotando 
    plt.figure().suptitle(f'${metodo_integracao} + ${metodo_otimizacao}')
    plt.plot(dados_P['tempo'], dados_P['concentração'], 'o', color='b', label="P exp")
    plt.plot(dados_S['tempo'], dados_S['concentração'], 'o', color='r', label='S exp') 
    plt.plot(solve_ivp.t, S_solve_ivp, color='red', label='S ajuste')
    plt.plot(solve_ivp.t, X_solve_ivp, 'o', color='green', label='X ajuste')
    plt.plot(solve_ivp.t, P_solve_ivp, 'o', color='blue', label='P ajuste')
    plt.xlabel(r"t - dias")
    plt.ylabel(r"Y - $kgDQO{m^3}$")
    plt.legend()
    plt.show()
        
def main():
    parametrosDadosXlsx = [0, 240, 25, 3] #tempo inicial, tempo final, número de pontos, algarismos significativos
    dados_P = ajustarXlsx("./xlsx1/produto.xlsx", parametrosDadosXlsx)
    dados_S = ajustarXlsx("./xlsx1/substrato.xlsx", parametrosDadosXlsx)
    #print(f'Substrato: \n ${dados_S}')  
    #print(f'Produto: \n ${dados_P}')  

     #condições iniciais das variáveis dos balanços: substrato, biomassa, produto;
    x = [42.5, 25.2, 0.0]
     #intervalo de integração, max_step e dados no tempo a serem retornados pela função integrate.solve_ivp:
    t_step = [0, 240]
    t_solve_ivp = dados_P['tempo']
    metodosIntegracao = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
    metodoIntegracao = metodosIntegracao[0]
    rtol = 3e-14
    atol = 1e-18
    
     ## Definindo parâmetros para Curve Fitting:
    paras = Parameters()
    paras.add('S_in', value=0., vary=False)
    paras.add('mumax_X', value=0.002, min=0.0001, max=0.015)
    paras.add('K_S', value=3.5, min=0.1, max=4)
    paras.add('Y_X_S', value=0.1, min=0, max=1)
    paras.add('Y_P_S', value=0.7, min=0.1, max=1.5)
    paras.add('k_dec', value=0.01, min=0.001, max=1)
    paras.add('D', value=0., vary=False)
    
    metodoMinimizacao = 'Nelder-Mead'
    
     ## Otimização para Substrato (0), Biomassa (1), Produto (2):
    indice = [0, 1, 2]
    dadoOtimizacao = [dados_S['concentração'], dados_P['concentração']]
    
    # print("minimize chamada para produto")
    # resultProduto = minimize(residual, paras, args=(t_step, dadoOtimizacao[1], t_solve_ivp, x, rtol, atol, indice[2], metodoIntegracao), method=metodoMinimizacao)  # leastsq
    # #report_fit(resultProduto) 
    # #execução da função de integração e plotagem do gráfico: 
    # print("plotagem chamada para produto")
    # plotagem(integracao(metodoIntegracao, t_step, resultProduto.params, t_solve_ivp, x, rtol, atol), dados_P, dados_S)
    
    print(chalk.red("minimize chamada para substrato"))
    resultSubstrato = minimize(residual, paras, args=(t_step, dadoOtimizacao[0], t_solve_ivp, x, rtol, atol, indice[0], metodoIntegracao), method=metodoMinimizacao)  # leastsq
    #report_fit(resultProduto) 
    report_fit(resultSubstrato)
    #execução da função de integração e plotagem do gráfico: 
    print(chalk.green("plotagem chamada para Substrato"))
    plotagem(integracao(metodoIntegracao, t_step, resultSubstrato.params, t_solve_ivp, x, rtol, atol), dados_P, dados_S, metodoIntegracao, metodoMinimizacao)
    
    
main()