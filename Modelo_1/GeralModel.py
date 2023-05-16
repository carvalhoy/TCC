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

    S = x[0] #g_DQO_S/m^3
    X = x[1] #g_DQO_X/m^3
    P = x[2] #g_DQO_P/m^3

    try:
         ##parâmetros do modelo
        S_in = params['S_in'].value
        mu_max_X = params['mumax_X'].value #dia^-1
        K_S = params['K_S'].value #g_DQO_S/m^3
        Y_X_S = params['Y_X_S'].value #g_DQO_X/g_DQO_S
        Y_P_S = params['Y_P_S'].value #g_DQO_P/g_DQO_S
        k_dec = params['k_dec'].value #dia^-1
        D = params['D'].value #dia^-1
    
    except KeyError:
        S_in, mu_max_X, K_S, Y_X_S, Y_P_S, k_dec, D = params
    
     ##definindo a reação:
    mu = mu_max_X*S/(K_S + S) #dia^-1
    
     ##definindo balanço de componentes:
    dS_dt = (D)*(S_in - S) - (1/Y_X_S)*mu*X
    dX_dt = (D)*(-X) + (mu-k_dec)*X
    dP_dt = (D)*(-P) + Y_P_S*((1/Y_X_S)*mu*X)

    return [dS_dt, dX_dt, dP_dt]

 ##ajustar dados utilizados na otimização:
def ajustarXlsx(caminho, parametrosDadosXlsx):
    [tempoInicial, tempoFinal, totalPontos, digitosDecimais] = parametrosDadosXlsx
    dados = pd.read_excel(caminho, header=None, names=['tempo', 'concentração'], decimal=',')
    dados['tempo'] = np.linspace(tempoInicial, tempoFinal, totalPontos)
    dados['concentração'] = round(dados['concentração']/1000, digitosDecimais)
    return dados

def residual(paras, t_step, data, t_solve_ivp, x, rtol, atol, indice, metodoIntegracao):
     ## resolução do modelo: 
    model = integracao(metodoIntegracao, t_step, paras, t_solve_ivp, x, rtol, atol)
     ## conversão de matriz numpy em dataframe pd:
    model = pd.DataFrame(model.y).transpose()
     ## definição do variável otimizada com indexação .iloc:
    modelVariaveldoModelo = model.iloc[:, indice]
     ## cálculo do erro entre dados experimentais e o ajuste do modelo:
    error = (modelVariaveldoModelo - data).ravel()
    return error

def residual2(paras, t_step, data, t_solve_ivp, x, rtol, atol, metodoIntegracao):
     ## resolução do modelo
    model = integracao(metodoIntegracao, t_step, paras, t_solve_ivp, x, rtol, atol)
     ## conversão de matriz numpy em dataframe pd:
    model = pd.DataFrame(model.y).transpose()
     ## definição da lista de resutados do modelo com indexação .iloc:
    model_S = model.iloc[:, 0] #substrato
    model_P = model.iloc[:, 2] #produto
     ## conversão da lista de listas de dados experimentais para dataframe pd:
    data = pd.DataFrame(data).transpose()
     ## cálculo do erro entre dados experimentais e ajuste do modelo:
    error = ((model_P - data.iloc[:, 1]) + (model_S - data.iloc[:, 0])).ravel()
    return error

def integracao(metodoIntegracao, t_step, params, t_solve_ivp, x, rtol, atol):
     ## integração do modelo com função .solve_ivp:
    solve_ivp = scipy.integrate.solve_ivp(model, t_step, x, metodoIntegracao, t_eval=t_solve_ivp, args=[params], rtol=rtol, atol=atol)
    return solve_ivp
    
def plotagem(solve_ivp, dados_P, dados_S, metodo_integracao, metodo_otimizacao):
     ## abrindo a tupla retornada em três variáveis:
    S_solve_ivp, X_solve_ivp, P_solve_ivp = solve_ivp.y    
     ## definindo título da figura:
    plt.figure().suptitle(f'${metodo_integracao} + ${metodo_otimizacao}')    
     ## plotando dados experimentais de concentração de produto:
    plt.plot(dados_P['tempo'], dados_P['concentração'], 'o', color='b', label="P exp")    
     ## plotando dados experimentais de concentração de substrato:
    plt.plot(dados_S['tempo'], dados_S['concentração'], 'o', color='r', label='S exp')     
     ## plotando resultado de S da integração do modelo:
    plt.plot(solve_ivp.t, S_solve_ivp, color='red', label='S ajuste')    
     ## plotando resultado de X da integração do modelo:
    plt.plot(solve_ivp.t, X_solve_ivp, color='green', label='X ajuste')    
     ## plotando resultado de P da integração do modelo:
    plt.plot(solve_ivp.t, P_solve_ivp, color='blue', label='P ajuste')    
     ## título eixo x:
    plt.xlabel(r"t - dias")    
     ## título eixo y:
    plt.ylabel(r"Y - $kgDQO{m^3}$")    
     ## chamada de legenda, localização da legenda fora do gráfico e tamanho da fonte:
    plt.legend(bbox_to_anchor=(1.05, 1.0) ,fontsize='8')   
     ## ajuste de espaçamento do gráfico:
    plt.tight_layout()    
     ## segurar a exibição das figuras até o final de todos os ajustes:
    plt.draw()
        
def main():
    parametrosDadosXlsx = [0, 240, 25, 3] #tempo inicial, tempo final, número de pontos, algarismos significativos
    
     ## importando dados experimentais do excel:
     ### checar caminho do arquivo###
    dados_P = ajustarXlsx("./xlsx1/produto.xlsx", parametrosDadosXlsx)
    dados_S = ajustarXlsx("./xlsx1/substrato.xlsx", parametrosDadosXlsx)
     ## condições iniciais das variáveis dos balanços: substrato, biomassa, produto;
    x = [42.5, 25.2, 0.0]
     ## intervalo de integração:
    t_step = [0, 240]
     ## dados no tempo a serem retornados pela função integrate.solve_ivp:
    t_solve_ivp = dados_P['tempo']
     ## lista de métodos de integração da função .solve_ivp:
    metodosIntegracao = ['RK45', 'RK23', 'DOP853', 'Radau', 'BDF', 'LSODA']
     ## definindo método de integração da função .solve_ivp:
    metodoIntegracao = metodosIntegracao[0]
     ## definindo as tolerâncias relativa e absoluta usada na função .solve_ivp:
    rtol = 3e-14
    atol = 1e-18    
    
     ## definindo parâmetros para Curve Fitting:
    paras = Parameters()
    paras.add('S_in', value=0., vary=False)
    paras.add('mumax_X', value=8.36, min=0.182, max=15.532)
    paras.add('K_S', value=0.326, min=0.0225, max=0.630)
    paras.add('Y_X_S', value=0.18, min=0.14, max=0.22)
    paras.add('Y_P_S', value=0.5, min=0.1, max=1.)
    paras.add('k_dec', value=0.045, min=0.04, max=0.05)
    paras.add('D', value=0., vary=False)   
     
     ## definindo método de minimização usado na função .minimize:
    metodoMinimizacao = 'Nelder-Mead'    
     ## otimização para Substrato (0), Biomassa (1), Produto (2):
    indice = [0, 1, 2]
     ## lista de listas com concentração de Substrato (0) e Produto (1)
    dadoOtimizacao = [dados_S['concentração'], dados_P['concentração']]
    
    print("minimize chamada para produto")
    resultProduto = minimize(residual, paras, args=(t_step, dadoOtimizacao[1], t_solve_ivp, x, rtol, atol, indice[2], metodoIntegracao), method=metodoMinimizacao)  
    report_fit(resultProduto)
    
     ## execução da função de integração otimizada para produto e plotagem do gráfico:
    print("plotagem chamada para produto")
    plotagem(integracao(metodoIntegracao, t_step, resultProduto.params, t_solve_ivp, x, rtol, atol), dados_P, dados_S, metodoIntegracao, metodoMinimizacao)
    
    print(chalk.red("minimize chamada para substrato"))
    resultSubstrato = minimize(residual, paras, args=(t_step, dadoOtimizacao[0], t_solve_ivp, x, rtol, atol, indice[0], metodoIntegracao), method=metodoMinimizacao)  
    report_fit(resultSubstrato)
    
     ## execução da função de integração otimizada para produto e plotagem do gráfico:
    print(chalk.green("plotagem chamada para Substrato"))
    plotagem(integracao(metodoIntegracao, t_step, resultSubstrato.params, t_solve_ivp, x, rtol, atol), dados_P, dados_S, metodoIntegracao, metodoMinimizacao)
    
    print(chalk.blue("minimize chamada para duas variáveis"))
    resultGeral = minimize(residual2, paras, args=(t_step, dadoOtimizacao, t_solve_ivp, x, rtol, atol, metodoIntegracao), method=metodoMinimizacao)  # leastsq
    report_fit(resultGeral)
    
     ## execução da função de integração e plotagem do gráfico: 
    print(chalk.green("plotagem chamada para duas variáveis"))
    plotagem(integracao(metodoIntegracao, t_step, resultGeral.params, t_solve_ivp, x, rtol, atol), dados_P, dados_S, metodoIntegracao, metodoMinimizacao)
    plt.show()
    
    
main()