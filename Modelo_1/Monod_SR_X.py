import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import tools
import models
import otimizador_Gouveia
import random

# Dados experimentais:
parametrosDadosXlsx:list[int] = [0, 240, 25, 3]
data_fit_P = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_produto.csv", parametrosDadosXlsx)
data_fit_S = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_substrato.csv", parametrosDadosXlsx)
data_fit_I = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_AGV.csv", parametrosDadosXlsx)
data = [data_fit_S,  data_fit_P]
t_exp = data_fit_P['tempo']

paras = lm.Parameters()
paras.add('S_in', value=0., vary=False) #kgDQO_S/m3
paras.add('umax', value=0.2, min=0.02, max=1.4) #dia-1
paras.add('Ks', value=100, min=0) #kgDQO_S/m3
paras.add('Yxs', 0.1, min=0. ) #kgDQO_X/kgDQO_S
paras.add('Yps', value=0.934, vary=False) #kgDQO_P/kgDQO_S
paras.add('kd',value=0.003, min=0.00) #dia-1
paras.add('D', value=0., vary=False) #dia-1
paras.add('S_0', value=38.2, vary=False)
paras.add('X_0', value=25.2, vary=False)
paras.add('P_0', value=0.,  vary=False)



condicoes_iniciais = [paras['S_0'].value, paras['P_0'].value]
simulação = models.model_monod_sr_x([0, 240], condicoes_iniciais, paras, None, False, None)
simu_S, simu_X, simu_P, simu_t = simulação


r2_simu = models.model_monod_sr_x([0, 240], condicoes_iniciais, paras, t_exp, True, data)

plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X')
plt.plot(simu_t, simu_P, 'r', label=f'P $(R^2={r2_simu[1]:.3f})$')
plt.legend(fontsize=15, loc='center right')
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()


############### Estimação de parâmetros:


sol_otim_P = lm.minimize(otimizador_Gouveia.obj_P_model_monod_sr_x, paras, 'leastsq', args=(data, [0, 240], condicoes_iniciais, t_exp))

print('\nResultado da otimização para a variável P\n')

print(f'Valor da função objetivo para os parâmetros otimizados: {np.sum(np.square(sol_otim_P.residual))}\n')
lm.report_fit(sol_otim_P)

simu_ajuste_P = models.model_monod_sr_x([0, 240], condicoes_iniciais, sol_otim_P.params, None, False, None)
simu_S, simu_X, simu_P, simu_t = simu_ajuste_P


r2_simu_ajuste_P = models.model_monod_sr_x([0, 240], condicoes_iniciais, sol_otim_P.params, t_exp, True, data)

# plt.title('Ajuste P')
plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu_ajuste_P[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X')
plt.plot(simu_t, simu_P, 'r', label=f'P $(R^2={r2_simu_ajuste_P[1]:.3f})$')
plt.legend(fontsize=15, loc='center right')
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()



############ Estimação de parâmetros multivariada:



sol_otim_SP = lm.minimize(otimizador_Gouveia.obj_SP_model_monod_sr_x, paras, 'leastsq', args=(data, [0, 240], condicoes_iniciais, t_exp), nan_policy='omit')

print('\nResultado da otimização multivariada SP\n')

print(f'Valor da função objetivo para os parâmetros otimizados: {np.sum(np.square(sol_otim_SP.residual))}\n')
lm.report_fit(sol_otim_SP)

simu_ajuste_SP = models.model_monod_sr_x([0, 240], condicoes_iniciais, sol_otim_SP.params, None, False, None)
simu_S, simu_X, simu_P, simu_t = simu_ajuste_SP


r2_simu_ajuste_SP = models.model_monod_sr_x ([0, 240], condicoes_iniciais, sol_otim_SP.params, t_exp, True, data)

# plt.title('Ajuste SP')
plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu_ajuste_SP[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X')
plt.plot(simu_t, simu_P, 'r', label=f'P $(R^2={r2_simu_ajuste_SP[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()
