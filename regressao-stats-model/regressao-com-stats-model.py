import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import scipy.stats as stats
import seaborn as sns

base = pd.read_csv('regressao-stats-model/mt_cars.csv') #importa o dataframe
print(base.shape)      # irá mostrar que o arquivo possui 32 linhas e 12 colunas
print(base.head())    # irá mostrar a matriz com os DADOS CATEGÓRICOS. 


# Para maior precisão, vamos excluir a coluna de dados categóricos.

base = base.drop(['Unnamed: 0'], axis=1)

print(base.shape)  # agora (32, 11) 
print(base.head())  

corr = base.corr() # calcula a correlação entre todas as colunas 

plt.figure(figsize=(8,6))  # garante que o heatmap fique separado
sns.heatmap(corr, cmap='coolwarm', annot=True, fmt='.2f') # com a biblioteca heatmap, desenha um mapa de calor da matriz de correlação:
plt.show() # exibe o gráfico na tela

#criando vários gráficos de dispersão (scatter plots) para analisar como o mpg se comporta em relação a outras variáveis.

column_pairs = [
    ('mpg', 'cyl'),
    ('mpg', 'disp'),
    ('mpg', 'hp'),
    ('mpg', 'wt'),
    ('mpg', 'drat'),
    ('mpg', 'vs')
] #escolha das variáveis que eu desejo comparar em relação a MPG 

n_plots = len(column_pairs) #a criação de gráficos será equivalente ao número de correlações que eu inseri em 'column_pairs'

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
axes = axes.flatten()

for i, pair in enumerate(column_pairs):
    x_col, y_col = pair
    sns.scatterplot(x=x_col, y=y_col, data=base, ax=axes[i])
    axes[i].set_title(f'{x_col} vs {y_col}') #O for percorre a lista de pares de colunas. Em cada iteração, ele pega um par, separa em x_col e y_col, e usa esses valores para criar um gráfico de dispersão.

plt.tight_layout() #organiza gráficos
plt.show() #exibição dos gráficos