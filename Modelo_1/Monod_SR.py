import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lmfit as lm
import tools
import models
import otimizador_Gouveia

# Dados experimentais:
parametrosDadosXlsx:list[int] = [0, 240, 25, 3]
data_fit_P = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_produto.csv", parametrosDadosXlsx)
data_fit_S = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_substrato.csv", parametrosDadosXlsx)
data_fit_I = tools.ajustarXlsx("../xlsx1/dados_gouveia_rao_AGV.csv", parametrosDadosXlsx)
data = [data_fit_S,  data_fit_P]
t_exp = data_fit_P['tempo']

paras = lm.Parameters()
paras.add('S_in', value=0., vary=False) #kgDQO_S/m3
paras.add('umax', value=0.05, min=0.02, max=0.8 ) #dia-1
paras.add('Ks', value=125.0, min=40.3, max=403.) #kgDQO_S/m3
paras.add('Yxs', value=0.07, min=0.01, max=1-0.877) #kgDQO_X/kgDQO_S
paras.add('Yps', value=0.877, vary=False) #kgDQO_P/kgDQO_S
paras.add('kd',value=0.001755, min=0.0005, max=0.035) #dia-1
paras.add('D', value=0., vary=False) #dia-1


condicoes_iniciais = [38.2, 25.2, 0.]
simulação = models.model_monod ([0, 240], condicoes_iniciais, paras, None, False, None)
simu_S, simu_X, simu_P, simu_t = simulação


r2_simu = models.model_monod ([0, 240], condicoes_iniciais, paras, t_exp, True, data)

plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X $(R^2={r2_simu[1]:.3f})$')
plt.plot(simu_t, simu_P, 'r', label=f'P')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()

# Estimação de parâmetros:

sol_otim_P = lm.minimize(otimizador_Gouveia.obj_model_monod_P, paras, 'leastsq', args=(data, [0, 240], condicoes_iniciais, t_exp))

print('\nResultado da otimização para avariável P\n')
lm.report_fit(sol_otim_P)

simu_ajuste_P = models.model_monod([0, 240], condicoes_iniciais, sol_otim_P.params, None, False, None)
simu_S, simu_X, simu_P, simu_t = simu_ajuste_P


r2_simu_ajuste_P = models.model_monod ([0, 240], condicoes_iniciais, sol_otim_P.params, t_exp, True, data)

plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu_ajuste_P[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X $(R^2={r2_simu_ajuste_P[1]:.3f})$')
plt.plot(simu_t, simu_P, 'r', label=f'P')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()

# Estimação de parâmetros multivariada:

sol_otim_SP = lm.minimize(otimizador_Gouveia.obj_model_monod_SP, sol_otim_P.params, 'leastsq', args=(data, [0, 240], condicoes_iniciais, t_exp))

print('\nResultado da otimização multivariada SP\n')

lm.report_fit(sol_otim_SP)

simu_ajuste_P = models.model_monod([0, 240], condicoes_iniciais, sol_otim_SP.params, None, False, None)
simu_S, simu_X, simu_P, simu_t = simu_ajuste_P


r2_simu_ajuste_P = models.model_monod ([0, 240], condicoes_iniciais, sol_otim_P.params, t_exp, True, data)

plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu_ajuste_P[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X')
plt.plot(simu_t, simu_P, 'r', label=f'P $(R^2={r2_simu_ajuste_P[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()
