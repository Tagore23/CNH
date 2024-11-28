import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Dados com listas de tamanhos diferentes
data = {
    'Capacidade Operacional': [8, 8, 9, 7, 8, 7, 7, 8, 9, 8, 9, 8, 8, 9, 7, 8, 10, 9, 9, 8, 10, 8, 7, 9, 10, 8, 9, 8, 8, 10, 8, 8, 8, 9, 9, 10, 8, 10, 10, 7, 9, 10, 10, 10, 10, 9, 8],
    'Adequação às Diversas Operações': [8, 8, 9, 8, 7, 8, 8, 7, 9, 9, 9, 8, 6, 9, 7, 10, 10, 8, 7, 8, 8, 8, 9, 8, 8, 9, 9, 9, 10, 9, 8, 8, 8, 9, 8, 10, 9, 10, 10, 9, 6, 9, 9, 10, 10, 8, 9],
    'Facilidade de Operação': [9, 8, 8, 8, 8, 8, 8, 7, 7, 9, 8, 7, 7, 7, 8, 8, 7, 9, 8, 9, 9, 7, 9, 7, 10, 8, 9, 7, 7, 10, 8, 10, 9, 9, 9, 9, 9, 10, 9, 8, 9, 8, 7, 8, 10, 8, 9],
    'Conforto e Ergonomia': [9, 8, 9, 8, 8, 8, 10, 9, 9, 7, 10, 9, 10, 8, 9, 8, 8, 8, 8, 9, 8, 9, 10, 10, 8, 9, 8, 8, 8, 10, 8, 9, 9, 8, 9, 10, 10, 10, 9, 9, 9],
    'Disponibilidade e Confiabilidade': [8, 8, 7, 5, 7, 7, 6, 7, 7, 9, 8, 7, 7, 6, 7, 9, 10, 8, 7, 6, 6, 8, 8, 7, 7, 7, 6, 6, 4, 7, 8, 8, 8, 7, 8, 7, 7, 9, 10, 8, 8],
    'Facilidade para Realização de Manutenções': [8, 8, 6, 6, 8, 7, 7, 7, 8, 8, 9, 7, 8, 7, 8, 2, 8, 7, 8, 10, 7, 9, 9, 8, 8, 9, 8, 8, 8, 4, 7, 7, 7, 9, 10, 8, 7, 10, 10, 10, 8],
    'Custo de Manutenção': [7, 8, 2, 3, 7, 6, 1, 7, 7, 7, 7, 7, 5, 8, 8, 8, 6, 6, 2, 8, 7, 8, 8, 9, 6, 10, 8, 10, 8, 7],
    'Consumo de Combustível': [6, 8, 8, 7, 5, 6, 7, 7, 8, 7, 7, 7, 6, 8, 5, 8, 7, 8, 6, 8, 10, 8, 8, 6, 8, 8, 8, 5, 8, 7, 7, 7, 9, 9, 8, 7, 8, 9, 9, 10],
    'Adaptabilidade': [8, 9, 8, 8, 6, 7, 8, 7, 8, 8, 9, 7, 9, 8, 7, 9, 10, 9, 7, 8, 8, 9, 8, 9, 9, 8, 8, 9, 5, 7, 8, 8, 8, 9, 9, 10, 7, 8, 10],
    'Facilidade de Uso do Piloto Automático': [9, 8, 8, 8, 7, 7, 8, 7, 6, 9, 7, 7, 4, 7, 5, 10, 6, 8, 10, 10, 7, 9, 9, 9, 8, 8, 8, 8, 6, 8, 8, 8, 9, 8, 10, 7, 8, 10],
    'Geração e Transmissão de Dados para Gestão da Frota': [8, 8, 0, 8, 5, 7, 6, 7, 8, 7, 9, 3, 5, 6, 5, 8, 10, 10, 8, 8, 8, 7, 8, 9, 10, 10, 10, 9, 9, 10, 8, 10, 10],
    'Geração e Transmissão de Dados para Gestão Agrícola': [8, 8, 0, 8, 6, 7, 6, 7, 7, 8, 9, 3, 5, 6, 5, 9, 9, 10, 8, 8, 9, 7, 8, 8, 10, 10, 6, 0, 10, 10, 9, 9, 9]
}

# Encontrando o comprimento máximo entre todas as listas
max_len = max(len(lst) for lst in data.values())

# Preenchendo as listas com NaN para que todas tenham o mesmo comprimento
for key in data:
    while len(data[key]) < max_len:
        data[key].append(np.nan)

# Criando o DataFrame
df = pd.DataFrame(data)

# Exibindo as primeiras linhas do DataFrame para verificar
print(df.head())

# Escalando os dados
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.fillna(df.mean()))  # Preenchendo NaN com a média de cada coluna

# Aplicando KMeans
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(scaled_data)

# Adicionando as predições (rótulos) ao DataFrame
df['Cluster'] = kmeans.labels_

# Plotando os clusters
plt.figure(figsize=(10, 6))
plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['Cluster'], cmap='viridis')
plt.title('Clusters K-Means Neighbors')
plt.xlabel('Capacidade Operacional')
plt.ylabel('Adequação às Diversas Operações')
plt.colorbar(label='Cluster')
plt.show()
