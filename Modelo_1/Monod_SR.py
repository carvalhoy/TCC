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

new_data_imtrying = pd.read_csv("../xlsx1/dados_biogas_cumulativo_rao.csv", header=None, names=['tempo', 'concentração'])

new_data_imtrying['tempo'] = round(new_data_imtrying['tempo'], 4) 

print(new_data_imtrying)
data = [data_fit_S,  data_fit_P]
t_exp = data_fit_P['tempo']

# paras = lm.Parameters()
# paras.add('S_in', value=0., vary=False) #kgDQO_S/m3
# paras.add('umax', value=0.05, min=0.0) #dia-1
# paras.add('Ks', value=0.7, min=0) #kgDQO_S/m3
# paras.add('Yxs', value=0.000000001, min=0.00, max=1) #kgDQO_X/kgDQO_S
# paras.add('Yps', value=0.877, min=0, max=1) #kgDQO_P/kgDQO_S
# paras.add('kd',value=0.001, min=0.00) #dia-1
# paras.add('D', value=0., vary=False) #dia-1

paras = lm.Parameters()
paras.add('S_in', value=0., vary=False) #kgDQO_S/m3
paras.add('umax', value=0.2, min=0.02, max=1.4) #dia-1
paras.add('Ks', value=100, min=0) #kgDQO_S/m3
paras.add('Yxs', value=0.1, min=0) #kgDQO_X/kgDQO_S
paras.add('Yps', value=0.934, vary=False) #kgDQO_P/kgDQO_S
paras.add('kd',value=0.003, min=0) #dia-1
paras.add('D', value=0., vary=False) #dia-1


condicoes_iniciais = [38.2, 25.2, 0.]
simulação = models.model_monod_sr ([0, 240], condicoes_iniciais, paras, None, False, None)
simu_S, simu_X, simu_P, simu_t = simulação


r2_simu = models.model_monod_sr ([0, 240], condicoes_iniciais, paras, t_exp, True, data)

plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X')
plt.plot(simu_t, simu_P, 'r', label=f'P $(R^2={r2_simu[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()


########## Estimação de parâmetros:

sol_otim_P = lm.minimize(otimizador_Gouveia.obj_model_monod_P, paras, 'leastsq', args=(data, [0, 240], condicoes_iniciais, t_exp))

print('\nResultado da otimização para a variável P\n')
lm.report_fit(sol_otim_P)

simu_ajuste_P = models.model_monod_sr([0, 240], condicoes_iniciais, sol_otim_P.params, None, False, None)
simu_S, simu_X, simu_P, simu_t = simu_ajuste_P


r2_simu_ajuste_P = models.model_monod_sr ([0, 240], condicoes_iniciais, sol_otim_P.params, t_exp, True, data)

plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu_ajuste_P[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X')
plt.plot(simu_t, simu_P, 'r', label=f'P $(R^2={r2_simu_ajuste_P[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()



############ Estimação de parâmetros multivariada:

paras_SP = lm.Parameters()
paras_SP.add('S_in', value=0., vary=False) #kgDQO_S/m3
paras_SP.add('umax', value=0.2, min=0.047, max=1.4) #dia-1
paras_SP.add('Ks', value=100, min=0) #kgDQO_S/m3
paras_SP.add('Yxs', value=0.1, min=0) #kgDQO_X/kgDQO_S
paras_SP.add('Yps', value=0.934, vary=False) #kgDQO_P/kgDQO_S
paras_SP.add('kd',value=0.003, min=0) #dia-1
paras_SP.add('D', value=0., vary=False) #dia-1


sol_otim_SP = lm.minimize(otimizador_Gouveia.obj_model_monod_SP, paras_SP, 'leastsq', args=(data, [0, 240], condicoes_iniciais, t_exp))

print('\nResultado da otimização multivariada SP\n')

lm.report_fit(sol_otim_SP)

simu_ajuste_SP = models.model_monod_sr([0, 240], condicoes_iniciais, sol_otim_SP.params, None, False, None)
simu_S, simu_X, simu_P, simu_t = simu_ajuste_SP


r2_simu_ajuste_SP = models.model_monod_sr ([0, 240], condicoes_iniciais, sol_otim_SP.params, t_exp, True, data)

plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu_ajuste_SP[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X')
plt.plot(simu_t, simu_P, 'r', label=f'P $(R^2={r2_simu_ajuste_SP[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()




### Estimação de parâmetros por ajuste de S nos primeiros instantes do processo (metanogênese é a etapa limitante):


paras = lm.Parameters()
paras.add('S_in', value=0., vary=False) #kgDQO_S/m3
paras.add('umax', value=0.04, vary=False) #dia-1
paras.add('Ks', value=7.9, min=1e-6) #kgDQO_S/m3
paras.add('Yxs', value=0.1, min=0) #kgDQO_X/kgDQO_S
paras.add('Yps', value=0.877, min=0) #kgDQO_P/kgDQO_S
paras.add('kd',value=0.002, min=0, max=0.04) #dia-1
paras.add('D', value=0., vary=False) #dia-1

print(data)
data_monod = [data_fit_S[:][:6], data_fit_P[:][:6]]

sol_otim_S = lm.minimize(otimizador_Gouveia.obj_model_monod_S, paras, 'leastsq', args=(data_monod, [0, 50], condicoes_iniciais, t_exp[:6]))
lm.report_fit(sol_otim_S)


simu_ajuste_S = models.model_monod_sr([0, 50], condicoes_iniciais, sol_otim_S.params, None, False, None)
simu_S, simu_X, simu_P, simu_t = simu_ajuste_S

r2_simu_ajuste_S = models.model_monod_sr ([0, 50], condicoes_iniciais, sol_otim_S.params, t_exp[:6], True, data_monod)


plt.plot(t_exp, data_fit_S['concentração'], 'bx', label='S exp.')
plt.plot(t_exp, data_fit_P['concentração'], 'rx', label='P exp.')
plt.plot(simu_t, simu_S, 'b', label=f'S $(R^2={r2_simu_ajuste_S[0]:.3f})$')
plt.plot(simu_t, simu_X, 'g', label=f'X')
plt.plot(simu_t, simu_P, 'r', label=f'P $(R^2={r2_simu_ajuste_S[1]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()
