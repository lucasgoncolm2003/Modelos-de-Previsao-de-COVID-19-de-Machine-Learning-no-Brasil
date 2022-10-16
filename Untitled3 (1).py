#!/usr/bin/env python
# coding: utf-8

# # Modelos de Previsão de COVID-19 de Machine Learning no Brasil por Regressão Logística de Prophet e Modelo ARIMA com Python e Jupyter - [Pandas, Numpy, Datetime, Plotly.Express, Plotly.Graph_Objects, Matplotlib, Seasonal_Decompose, Prophet & ARIMA]
# 
# ## Análise de Séries Temporais sobre a Contaminação da COVID-19.
# ### - Modelos de Previsão de Evolução da Pandemia da COVID-19 no Brasil
# ### - Linguagem Python e Plataforma Jupyter
# ### - Machine Learning ou Aprendizagem de Máquina
# #### - Decomposição de Séries Temporais com Modelo Auto-Regressivo Integrado de Médias Móveis (ARIMA)
# #### - Modelo de Crescimento com Regressão Logística para Predição de Função Sigmóide em Prophet

# In[24]:


import pandas as pd
import numpy as np
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go


# In[11]:


url = 'https://github.com/neylsoncrepalde/projeto_eda_covid/blob/master/covid_19_data.csv?raw=true'
# read_csv: faz Leitura da Tabela e converte ObservationDate e Last Update em Datas
df = pd.read_csv(url, parse_dates=['ObservationDate', 'Last Update'])
print(df)
df.dtypes
import re
# re.sub: substitui / por Espaço, deixando o Nome das Colunas em Lower
def corrige_colunas(col_name):
    return re.sub(r"[/| ]", "", col_name).lower()
# Substitui / por Espaço e coloca em Lower o nome de todas as Colunas
df.columns = [corrige_colunas(col) for col in df.columns]
print(df)


# In[12]:


# loc: Localiza os Registros em que o País seja o Brasil
df.loc[df.countryregion == 'Brazil']


# In[13]:


# Encontra os Registros do Brasil e com Confirmação Positiva
brasil = df.loc[(df.countryregion == 'Brazil') & (df.confirmed > 0)]
# line: esboço de Gráfico Linear de "Casos Confirmados no Brasil"
# Eixo x: nomeado Data, baseado na Variável observationdate
# Eixo y: nomeado Número de Casos Confirmados, baseado na variável confirmed
get_ipython().run_line_magic('pinfo', 'px.line')
px.line(brasil, 'observationdate', 'confirmed', 
        labels={'observationdate':'Data', 'confirmed':'Número de Casos Confirmados'},
       title='Casos Confirmados no Brasil')


# In[14]:


# Nova Coluna novoscasos igual a uma Lista de Map, que pega uma Função e aplica em todos os casos Confirmados
# Map retorna um Iterador em Python, sendo necessário colocar as Iterações em uma List
brasil['novoscasos'] = list(map(
    # Função Lambda para Programação de Funções Anônimas que itera por todos os Casos do Banco de Dados
    # retorna 0 se a Linha for 0, senão a Diferença do Dia Atual com o Anterior é informada.
    lambda x: 0 if (x==0) else brasil['confirmed'].iloc[x] - brasil['confirmed'].iloc[x-1],
    # iloc: faz Subset de Variáveis e Colunas pelo Índice.
    np.arange(brasil.shape[0])
    # shape: retorna as Dimensões do DataFrame, e se for [0], pega apenas a Primeira Informação: as Linhas da Coluna.
))


# In[ ]:


# Esboço de Gráfico Linear de "Novos Casos por Dia" com x como observationdate, y como novoscasos
# Rótulos de Data para x e Novos Casos para y.
px.line(brasil, x='observationdate', y='novoscasos', title='Novos Casos por Dia',
       labels={'observationdate': 'Data', 'novoscasos': 'Novos Casos'})


# In[15]:


# Atribui a fig uma Figura
fig = go.Figure()
# Adiciona Traço de Reta com Eixo x de Data, Eixo y de Número de Mortes
# Modo de Linhas com Círculos de Marcação
# Retas e Marcação de Cor Vermelha.
fig.add_trace(
    go.Scatter(x=brasil.observationdate, y=brasil.deaths, name='Mortes', mode='lines+markers',
              line=dict(color='red'))
)
# Edição de Layout: Títulos do Eixo x, Eixo y e do Gráfico Geral
fig.update_layout(title='Mortes por COVID-19 no Brasil',
                   xaxis_title='Data',
                   yaxis_title='Número de mortes')
fig.show()


# In[16]:


# Cálculo da Taxa de Crescimento: (Diapresente/Diapassado)^(1/ndias)-1
def taxa_crescimento(data, variable, data_inicio=None, data_fim=None):
    # Se data_inicio for None, define como a primeira data disponível no dataset
    if data_inicio == None:
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
        # Mínima Data onda há mais de 0 Casos
    else:
        data_inicio = pd.to_datetime(data_inicio)
        # Transforma a Data de Início em DateTime
    if data_fim == None:
        data_fim = data.observationdate.iloc[-1]
        # Última Data onde ocorreu um caso
    else:
        data_fim = pd.to_datetime(data_fim)
        # Transforma a Data de Fim em DateTime
    # Define os valores de presente e passado
    passado = data.loc[data.observationdate == data_inicio, variable].values[0]
    presente = data.loc[data.observationdate == data_fim, variable].values[0]
    # Define o Número de Pontos no tempo que se deseja avaliar
    n = (data_fim - data_inicio).days
    # Calcula a taxa
    taxa = (presente/passado)**(1/n) - 1
    return taxa*100
cresc_medio = taxa_crescimento(brasil, 'confirmed')
print(f"O Crescimento Médio do COVID no Brasil no Período Avaliado foi de {cresc_medio.round(2)}%.")


# In[17]:


# Taxa de Crescimento Diária
def taxa_crescimento_diaria(data, variable, data_inicio=None):
    if data_inicio == None:
    # Data de Início é a Data Mínima de Observação
        data_inicio = data.observationdate.loc[data[variable] > 0].min()
    else:
        data_inicio = pd.to_datetime(data_inicio)
    # Data de Fim é a Data Máxima de Observação
    data_fim = data.observationdate.max()
    n = (data_fim - data_inicio).days
    taxas = list(map(
        # Função Lambda: Atualização em Relação ao Dia Anterior: (Hoje - Ontem)/Ontem
        lambda x: (data[variable].iloc[x] - data[variable].iloc[x-1]) / data[variable].iloc[x-1],
        # Range de 1 até n+1, pois pega o Segundo Dia em diante, até n.
        range(1,n+1)
    ))
    return np.array(taxas)*100
tx_dia = taxa_crescimento_diaria(brasil, 'confirmed')
print(tx_dia)


# In[18]:


# Primeiro Dia é o Dia em que a Data de Observação é Mínima, com mais de 0 Casos Confirmados.
primeiro_dia = brasil.observationdate.loc[brasil.confirmed > 0].min()
# Gráfico de Linha de Taxa de Crescimento de Casos Confirmados com:
# Eixo x: Retorno de Dias do Primeiro Dia até o Último Dia, onde Observation Date é Máximo, do Segundo dia ao último [1:]
# Eixo y: Taxa de Crescimento Diária.
px.line(x=pd.date_range(primeiro_dia, brasil.observationdate.max())[1:],
        y=tx_dia, title='Taxa de Crescimento de Casos Confirmados no Brasil',
       labels={'y':'Taxa de Crescimento', 'x':'Data'})
# Rótulos de "Taxa de Crescimento" para Eixo y e de "Data" para Eixo x


# In[19]:


# ---- Machine Learning
from statsmodels.tsa.seasonal import seasonal_decompose # Biblioteca Seasonal_Decompose
import matplotlib.pyplot as plt # Biblioteca Matplotlib para Gráficos Estáticos
novoscasos = brasil.novoscasos
novoscasos.index = brasil.observationdate # Índices são Datas
# seasonal_decompose: faz a Decomposição dos novoscasos
# É preciso saber os Observados, a Tendência, a Sazonalidade e o Ruído
res = seasonal_decompose(novoscasos)
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
# 4 Linhas, 1 Coluna, e uma Figura de 10x8
ax1.plot(res.observed) # Eixo 1: o Eixo dos Observados
ax2.plot(res.trend) # Eixo 2: o Eixo das Tendências
ax3.plot(res.seasonal) # Eixo 3: o Eixo da Sazonalidade
ax4.scatter(novoscasos.index, res.resid) # Eixo 4: os Resíduos ou Ruídos
ax4.axhline(0, linestyle='dashed', c='black') # Linha Horizontal no 0 para Marcação
plt.show()


# In[ ]:


confirmados = brasil.confirmed
confirmados.index = brasil.observationdate # Índices são as Datas
res2 = seasonal_decompose(confirmados)
# seasonal_decompose: faz a Decomposição a Série dos confirmados
# É preciso saber os Observados, a Tendência, a Sazonalidade e o Ruído
fig, (ax1,ax2,ax3, ax4) = plt.subplots(4, 1,figsize=(10,8))
# 4 Linhas, 1 Coluna, e uma Figura de 10x8
ax1.plot(res2.observed) # Eixo 1: o Eixo dos Observados
ax2.plot(res2.trend) # Eixo 2: o Eixo das Tendências
ax3.plot(res2.seasonal) # Eixo 3: o Eixo da Sazonalidade
ax4.scatter(confirmados.index, res2.resid) # Eixo 4: o Eixo dos Resíduos
ax4.axhline(0, linestyle='dashed', c='black') # Linha Horizontal no 0 para Marcação
plt.show()


# In[20]:


get_ipython().system('pip install pmdarima')


# In[21]:


# Modelo de Séries Temporais - Média Móvel Integrada Auto-Regressiva
# ARIMA - AutoRegressive Integrated Moving Average
# Modelagem e Previsão do Futuro com base no Passado
from pmdarima.arima import auto_arima # Importação da Biblioteca ARIMA
modelo = auto_arima(confirmados) # Ajuste Automático da Melhor Modelagem ARIMA Possível
# Encontra automaticamente a Sazonalidade, Tendência, Observados e Ruídos
# data_range: coloca um Intervalo de Predição
pd.date_range('2020-05-01', '2021-12-01')


# In[ ]:


fig = go.Figure(go.Scatter(x=confirmados.index, y=confirmados, name='Observados'))
# add_trace: coloca Traço com Eixo x de Confirmados, e Eixo y como Predição
fig.add_trace(go.Scatter(x=confirmados.index, y = modelo.predict_in_sample(), name='Preditos'))
# predict_in_sample: faz o Aprendizado e o uso para prever a Série por cima da Linha de Observados
fig.add_trace(go.Scatter(x=pd.date_range('2020-05-20', '2020-06-20'), y=modelo.predict(31), name='Forecast'))
# predict(x): faz a Previsão de x dias.
# add_trace: coloca Traço com Eixo x de Intervalo, e Eixo y como Predição
# update_layout: Título do Gráfico e dos Eixos
fig.update_layout(title='Previsão de Casos Confirmados para os Próximos Meses',
                 yaxis_title='Casos Confirmados', xaxis_title='Data')
fig.show()


# In[ ]:


get_ipython().system('conda install -c conda-forge fbprophet -y')


# In[ ]:


# ---- Modelo de Crescimento: Biblioteca Prophet, do Facebook
from fbprophet import Prophet
# Preparação ou Pré-Processamento de Dados
train = confirmados.reset_index()[:-5] # do Início até antes dos 5 Últimos
test = confirmados.reset_index()[-5:] # Testa os 5 Últimos

# Renomeação de Colunas para Testes
train.rename(columns={"observationdate":"ds","confirmed":"y"},inplace=True) # ds e y é o Padrão da Biblioteca
test.rename(columns={"observationdate":"ds","confirmed":"y"},inplace=True) # inplace=True modifica os Dados de Train e Teste
test = test.set_index("ds")
test = test['y']
# Esboço de Modelo de Crescimento com base na Função Sigmóide de Contágio
# A Regressão Logística estima a Função Sigmóide, logo growth = "logistic"
# changepoints: Pontos de Mudanças Bruscas
profeta = Prophet(growth="logistic", changepoints=['2020-03-21', '2020-03-30', '2020-04-25', '2020-05-03', '2020-05-10'])
pop = 211463256 #População do Brasil segundo a Projeção do IBGE
train['cap'] = pop # A Capacidade do Treino abrange a População
profeta.fit(train) # Treinamento do Modelo de Treinamento

# Construindo previsões para o futuro
# Projeção de DataFrame de Predição de 200 Dias no Futuro
future_dates = profeta.make_future_dataframe(periods=200)
future_dates['cap'] = pop # a Capacidade de Future_Dates é para suportar a População
# Predição de Futuras Datas
forecast =  profeta.predict(future_dates) # Predição de Future_Dates


# In[ ]:


fig = go.Figure()
# Adicionar Reta baseada em Forecast.ds e Forecast.yhat chamada "Predição"
fig.add_trace(go.Scatter(x=forecast.ds, y=forecast.yhat, name='Predição'))
# Adicionar Reta baseada em Índices de Teste e Teste chamada "Observados - Teste"
# fig.add_trace(go.Scatter(x=test.index, y=test, name='Observados - Teste'))
# Adicionar Reta baseada em Datas do Treino por Confirmados chamada "Observados - Treino"
fig.add_trace(go.Scatter(x=train.ds, y=train.y, name='Observados - Treino'))
# Update em Layout para  Título
fig.update_layout(title='Predições de Casos Confirmados no Brasil')
fig.show()


# In[ ]:





# In[ ]:




