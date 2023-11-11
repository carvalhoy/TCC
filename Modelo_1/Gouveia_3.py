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
data = [data_fit_S, data_fit_P]

t = data_fit_P['tempo']

## Parâmetros do modelo otimizado de Gouvea et al. (2022):
paras = lm.Parameters()
paras.add('sAr_0', value = 38.1625*0.3532, vary=False)
paras.add('sAl_0', value=38.1625*(1-0.3532), vary=False)
paras.add('kl', value=0.0133, vary=False) #dia-1
paras.add('kr', value=0.1532, vary=False) #kgDQO_P/kgDQO_S
paras.add('k2', value=0.1274, vary=False) #kgDQO_X/kgDQO_S
paras.add('k3', value=0.1181, vary=False)

condicoes_iniciais = [paras['sAr_0'].value, paras['sAl_0'].value, 0., 0.]
sim = models.wrapper_model_2([0, 240], condicoes_iniciais, paras, None, False, None)
sAr, sAl, sB, sC, sD, t_sim, sAt = sim
# alfa_calc = sAr/sAt

R2_S_P = models.wrapper_model_2([0, 240], condicoes_iniciais, paras, t, True, data)

# Plotagem:
plt.plot(t_sim, sAr, 'c', label='$S_{A,R}$')
plt.plot(t_sim, sAl, 'aquamarine', label='$S_{A,L}$')
plt.plot(t_sim, sAt, 'b', label=f'$S_A (R^2={R2_S_P[0]:.3f})$')
plt.plot(t_sim, sB, 'y', label='$S_B$')
plt.plot(t_sim, sC, 'g', label='$S_C$')
plt.plot(t_sim, sD, 'r', label=f'$S_D (R^2={R2_S_P[1]:.3f})$')
plt.plot(t, data_fit_S['concentração'], 'bx', label='$S_A exp.$')
plt.plot(t, data_fit_P['concentração'], 'rx', label='$S_D exp.$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()

# Estimação de parametros:

paras = lm.Parameters()
paras.add('sAr_0', value = 38.1625*0.5,        min=0.)
paras.add('sAl_0', value = 38.1625*(1-0.5),    min=0.)
paras.add('kl',    value = 1.84702752e-02/2,   min=0.) #dia-1
paras.add('kr',    value = 1.84702752e-02,     min=0.) #kgDQO_P/kgDQO_S
paras.add('k2',    value = 1.84702752e-02 * 2, min=0) #kgDQO_X/kgDQO_S
paras.add('k3',    value = 1.84702752e-02 * 3, min=0)

# paras = lm.Parameters()
# paras.add('sAr_0', value = 38.1625*0.3532, vary=False)
# paras.add('sAl_0', value=38.1625*(1-0.3532), vary=False)
# paras.add('kl', value=0.0133, min=0) #dia-1
# paras.add('kr', value=0.1532, min=0) #kgDQO_P/kgDQO_S
# paras.add('k2', value=0.1274, min=0) #kgDQO_X/kgDQO_S
# paras.add('k3', value=0.1181, min=0)

condicoes_iniciais = [paras['sAr_0'].value, paras['sAl_0'].value, 0., 0.]
res_otimizacao = lm.minimize(otimizador_Gouveia.obj_wrapper_model_2, paras, 'leastsq', args=(data_fit_P['concentração'], [0, 240], condicoes_iniciais, t))

print('\nResultado da otimização de P\n')
lm.report_fit(res_otimizacao)

condicoes_iniciais_otimizadas = [res_otimizacao.params['sAr_0'].value, res_otimizacao.params['sAl_0'].value, 0., 0.]

sim_ajuste = models.wrapper_model_2([0, 240], condicoes_iniciais_otimizadas, res_otimizacao.params, None, False, None)
sAr, sAl, sB, sC, sD, t_sim, sAt = sim_ajuste
R2_S_P_ajuste = models.wrapper_model_2([0, 240], condicoes_iniciais_otimizadas, res_otimizacao.params, t, True, data)

# Plotagem do resultado da estimação de parametros:
plt.title('Simulação do ajuste de P')
plt.plot(t_sim, sAr, 'c', label='$S_{A,R}$')
plt.plot(t_sim, sAl, 'aquamarine', label='$S_{A,L}$')
plt.plot(t_sim, sAt, 'b', label=f'$S_A (R^2={R2_S_P_ajuste[0]:.3f})$')
plt.plot(t_sim, sB, 'y', label='$S_B$')
plt.plot(t_sim, sC, 'g', label='$S_C$')
plt.plot(t_sim, sD, 'r', label=f'$S_D (R^2={R2_S_P_ajuste[1]:.3f})$')
plt.plot(t, data_fit_S['concentração'], 'bx', label='$S_A exp.$')
plt.plot(t, data_fit_P['concentração'], 'rx', label='$S_D exp.$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()

# Estimação multivariável de parâmetros SP:

paras = lm.Parameters()
paras.add('sAr_0', value = 38.1625*0.3532,vary=False)
paras.add('sAl_0', value=38.1625*(1-0.3532),vary=False)
paras.add('kl', value=1.84702752e-02/2, min=0) #dia-1
paras.add('kr', value=1.84702752e-02, min=0) #kgDQO_P/kgDQO_S
paras.add('k2', value = 1.84702752e-02 * 2, min=0) #kgDQO_X/kgDQO_S
paras.add('k3', value=1.84702752e-02 * 3, min=0)

condicoes_iniciais = [paras['sAr_0'].value, paras['sAl_0'].value, 0., 0.]
data_multivar_SP = [data_fit_S, data_fit_P]
print(np.shape(data_multivar_SP))
res_otimizacao_multivar_SP = lm.minimize(otimizador_Gouveia.obj_wrapper_model_2_SP, paras, 'least_squares', args=(data_multivar_SP, [0, 240], condicoes_iniciais, t))

print('\nResultado da otimização multivariada de SP\n')
lm.report_fit(res_otimizacao_multivar_SP)

###condição inicial de SC fixada no valor extraído no WebPlotDigitizer
condicoes_init_otim_multivar_SP = [res_otimizacao_multivar_SP.params['sAr_0'].value, res_otimizacao_multivar_SP.params['sAl_0'].value, 0., 1.27]


sim_ajuste_multivar = models.wrapper_model_2([0, 240], condicoes_init_otim_multivar_SP, res_otimizacao_multivar_SP.params, None, False, None)
sAr, sAl, sB, sC, sD, t_sim, sAt = sim_ajuste_multivar
R2_S_P_ajuste_multivar = models.wrapper_model_2([0, 240], condicoes_init_otim_multivar_SP, res_otimizacao_multivar_SP.params, t, True, data_multivar_SP)

# Plotagem do resultado da estimação de parametros:
# plt.title('Simulação do ajuste multivariável SP')
plt.plot(t_sim, sAr, 'c', label='$S_{A,R}$')
plt.plot(t_sim, sAl, 'aquamarine', label='$S_{A,L}$')
plt.plot(t_sim, sAt, 'b', label=f'$S_A (R^2={R2_S_P_ajuste_multivar[0]:.3f})$')
plt.plot(t_sim, sB, 'y', label='$S_B$')
plt.plot(t_sim, sC, 'g', label=f'$S_C$')
plt.plot(t_sim, sD, 'r', label=f'$S_D (R^2={R2_S_P_ajuste_multivar[1]:.3f})$')
plt.plot(t, data_fit_S['concentração'], 'bx', label='$S_A$ exp.')
# plt.plot(t, data_fit_I['concentração'], 'gx', label='$S_C exp.$')
plt.plot(t, data_fit_P['concentração'], 'rx', label='$S_D$ exp.')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()


# Estimação multivariável de parâmetros SPI:

paras = lm.Parameters()
paras.add('sAr_0', value = 38.1625*0.3532,vary=False)
paras.add('sAl_0', value=38.1625*(1-0.3532),vary=False)
paras.add('kl', value=1.84702752e-02/2, min=0) #dia-1
paras.add('kr', value=1.84702752e-02, min=0) #kgDQO_P/kgDQO_S
paras.add('k2', value = 1.84702752e-02 * 2, min=0) #kgDQO_X/kgDQO_S
paras.add('k3', value=1.84702752e-02 * 3, min=0)

condicoes_iniciais = [paras['sAr_0'].value, paras['sAl_0'].value, 0., 0.]
data_multivar_SPI = [data_fit_S, data_fit_I, data_fit_P]
print(np.shape(data_multivar_SPI))
res_otimizacao_multivar_SPI = lm.minimize(otimizador_Gouveia.obj_wrapper_model_2_SPI, paras, 'least_squares', args=(data_multivar_SPI, [0, 240], condicoes_iniciais, t))

print('\nResultado da otimização multivariada de SPI\n')
lm.report_fit(res_otimizacao_multivar_SPI)

condicoes_init_otim_multivar_SPI = [res_otimizacao_multivar_SPI.params['sAr_0'].value, res_otimizacao_multivar_SPI.params['sAl_0'].value, 0., 1.27]


sim_ajuste_multivar_SPI = models.wrapper_model_2([0, 240], condicoes_init_otim_multivar_SPI, res_otimizacao_multivar_SPI.params, None, False, None)
sAr, sAl, sB, sC, sD, t_sim, sAt = sim_ajuste_multivar_SPI
R2_S_P_ajuste_multivar = models.wrapper_model_2([0, 240], condicoes_init_otim_multivar_SPI, res_otimizacao_multivar_SPI.params, t, True, data_multivar_SPI)

# Plotagem do resultado da estimação de parametros:
plt.title('Simulação do ajuste multivariável SPI')
plt.plot(t, data_fit_S['concentração'], 'bx', label='$S_A exp.$')
plt.plot(t, data_fit_I['concentração'], 'gx', label='$S_C exp.$')
plt.plot(t, data_fit_P['concentração'], 'rx', label='$S_D exp.$')
plt.plot(t_sim, sAr, 'c', label='$S_{A,R}$')
plt.plot(t_sim, sAl, 'aquamarine', label='$S_{A,L}$')
plt.plot(t_sim, sAt, 'b', label=f'$S_A (R^2={R2_S_P_ajuste_multivar[0]:.3f})$')
plt.plot(t_sim, sB, 'y', label='$S_B$')
plt.plot(t_sim, sC, 'g', label=f'$S_C (R^2={R2_S_P_ajuste_multivar[1]:.3f}$')
plt.plot(t_sim, sD, 'r', label=f'$S_D (R^2={R2_S_P_ajuste_multivar[2]:.3f})$')
plt.legend(fontsize=15)
plt.xlabel('t - dias', fontsize=15)
plt.ylabel('$kg_{DQO}/m^3$', fontsize=15)
plt.show()



    
    