import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
import pandas as pd

'''
Código para predição de velocidade de flutter utilizando o método K. 
Disciplina: Projeto Integrador de Engenharia Mecânica
Aluno: Marco Aurélio Rezende Silva
Orientador: Maurício Francisco Caliri Júnior
'''
# Função de Theodorsen
def fTh(k):
    if k <= 0.5:
            d1 = 1 - (0.045/k)*1j
            d2 = 1 - (0.3/k)*1j
            Ck = 1 - (0.165/d1) - (0.335/d2)
    else:
            d3 = 1 - (0.041/k)*1j 
            d4 = 1 - (0.32/k)*1j
            Ck = 1 - (0.165/d3) - (0.335/d4)

    return Ck

####################### INPUTS
# # Variáveis do sistema - teste 1  (Dados retirados do Fung)
# rho=1.225               # densidade do ar [kg/m³]
# b=0.127                 # semi corda do aerofólio [m]
# a=-0.15                 # localização do centro de cisalhamento/eixo elástico 
# mi=76                   # razão de massa [admensional]
# m=1                     # massa do aerofólio [kg]
# x_alpha=0.25            # posição do c.g em relação ao eixo elástico [admensional]
# r_alpha2=0.388          # raio de giração [admensional]
# omega_h=55.9            # frequência de flexão [rad/s]
# omega_alpha=64.1        # frequência de torção [rad/s]
# R=omega_h/omega_alpha   # razão de frequências [admensional]

# Variáveis do sistema - teste 2  (Dados retirados do Fung)
rho=1.225                 # densidade do ar [kg/m³]
b=9.144                   # semi corda do aerofólio [m]
a=0                       # localização do centro de cisalhamento/eixo elástico
mi=40                     # razão de massa [admensional]
m=3925                    # massa do aerofólio [kg]
x_alpha=0                 # posição do c.g em relação ao eixo elástico [admensional]
r_alpha2=0.622            # raio de giração [admensional]
omega_h=0.880             # frequência de flexão [rad/s]
omega_alpha=1.552         # frequência de torção [rad/s]
R=omega_h/omega_alpha     # razão de frequências [admensional]

# # Variáveis do sistema (Dados retirados do Bisplinghoff/Goland/Artigo)
# rho=1.225  #slugsd/ft3           # densidade do ar 0.0765*0.0005787 #/(144*12)#*
# l=6.096
# b= .9145                                 #feeti corda do aerofólio
# a= - .34#-0.17                         # localização do centro de cisalhamento/elastico  a_hpós meio da asa/corda
# m= 35.72 #35.72                            # massa do aerofólio/ comprimento de envergadura
# x_alpha= 0.2    # posição do cg em relacao eixo elástico pós meio da asa/corda tbm S*
# ialp = 7.452
# r_alpha2=ialp/(m*b**2)          # raio de giração I_alpha/m   I/m*b^2 (adm)
# mi=m/(np.pi*rho*b**2)                # razão de massa m/pi*rho*b**2
# omega_h=49.49
# omega_alpha=87.08
# R=omega_h/omega_alpha # razão de frequências



######################## ROTINA PARA CÁLCULO
# Frequências naturais - Vibração livre
M = np.array([[1 , x_alpha] , [x_alpha , r_alpha2]])        # matriz de massa do sistema
K = np.array([[R**2 , 0] , [0 , r_alpha2] ])                # matriz de rigidez do sistema
A,V = linalg.eig(K,M)                                       # extração de autovalores e autovetores generalizados das matrizes K e M

Mm = np.round(np.dot(np.dot(V.T,M),V), decimals=6)          # normalização da matriz de massa (domínio modal)
Km = np.round(np.dot(np.dot(V.T,K),V), decimals=6)          # normalização da matriz de rigidez (domínio modal)

# Cálculo da velocidade de flutter
ii = 0
var = np.arange(3,0.1,-1e-4)                                # faixa de variação da frequência reduzida 'k' (número de linhas da matriz)
var2 = [0, 1]                                               # número de colunas da matriz
aut = np.zeros((len(var),len(var2)),dtype=complex)          # definição da matriz de autovalores do sistema
vel = np.zeros((len(var),len(var2)))                        # definição da matriz de velocidade do sistema
g = np.zeros((len(var),len(var2)))                          # definição da matriz de amortecimento do sistema
omega = np.zeros((len(var),len(var2)))                      # definição da matriz de frequências do sistema

for k in var:                                               # condição de loop do sistema considerando os valores de k
    Ck = fTh(k)                                             # utilização da função de Theodorsen
    L_h = 1 - 1j*2*Ck/k                                     # sustentação referente ao modo de flexão
    L_alpha = 0.5 - 1j*(1+2*Ck)/k - 2*Ck/k**2               # sustentação referente ao modo de torção
    M_h = 0.5                                               # momento referente ao modo de flexão
    M_alpha = 3/8 - 1j/k                                    # momento referente ao modo de torção

    Q11 = (L_h)                                                             # termo A11 da matriz aerodinâmica
    Q12 = (L_alpha - (0.5 + a)*(L_h))                                       # termo A12 da matriz aerodinâmica
    Q21 = (M_h - (0.5 + a)*(L_h))                                           # termo A21 da matriz aerodinâmica
    Q22 = (M_alpha - (0.5 + a)*(L_alpha + M_h) + ((0.5 + a)**2)*(L_h))      # termo A22 da matriz aerodinâmica

    Q = np.array([[Q11 , Q12] , [Q21 , Q22]])/mi            # matriz aerodinâmica
    Qm = np.dot(np.dot(V.T,Q),V)                            # matriz aerodinâmica normalizada (domínio modal)

# Aplicação do método V-g
    v = np.sort(linalg.eigvals(Mm + Qm , Km))               # cálculo dos autovalores da matriz aeroelástica      
    aut[ii,0] = v[0]                                        # ordenando a matriz de autovalores (1° linha de v sendo 1° linha 1° coluna de aut)
    aut[ii,1] = v[1]                                        # ordenando a matriz de autovalores (2° linha de v sendo 1° linha 2° coluna de aut)

    for ll in var2:                                         
          r = np.real(aut[ii,ll])                           # separação da parte real do sistema
          im = np.imag(aut[ii,ll])                          # separação da parte imaginária do sistema
          vel[ii,ll] = np.sqrt(1/r)/k*omega_alpha*b         # cálculo da velocidade
          g[ii,ll] = im/r                                   # cálculo do amortecimento
          omega[ii,ll] = np.sqrt(1/r)*omega_alpha           # cálculo da frequência

    ii += 1     


# Apresentação dos resultados
aux1 = np.argwhere(g>0)                                     # encontrando os pontos onde o amortecimento se torna positivo (ponto de flutter)
vel_aux = vel[aux1[:,0],aux1[:,1]]                         # indexando a velocidade onde o amortecimento se torna positivo            
v_flutter = np.round(np.min(vel_aux),decimals = 2)          # extraindo o valor mínimo de velocidade onde o amortecimento é positivo (ponto da velocidade de flutter)
v_flutter_mph = np.round(v_flutter/1.465,decimals = 2)


print('Velocidade de flutter:',v_flutter)
print('Velocidade de flutter:',v_flutter_mph,'mph')

# Plotando os gráficos
plt.subplot(2,1,1)
plt.title('Diagrama VGF')
plt.plot(vel,omega,label = ['Modo 1','Modo 2'])
string_flutter = 'Velocidade de Flutter = ' + str(np.round(v_flutter,decimals=2))
plt.axvline(x=v_flutter,linestyle = '--',color = 'k',label = string_flutter )       # plotando linha onde o flutter acontece
plt.ylabel(r'$\omega \; \left[\frac{rad}{s}\right]$')
plt.legend()
plt.grid()

plt.subplot(2,1,2)
plt.plot(vel,g)
plt.axvline(x=v_flutter,linestyle = '--',color = 'k')
plt.xlabel('Velocidade (EAS) [m/s]')
plt.ylabel('Amortecimento')
plt.grid()

resolution_value = 1200
plt.savefig("teste3.png", format="png", dpi=resolution_value)

plt.show()