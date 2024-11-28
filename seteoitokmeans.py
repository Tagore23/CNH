import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Dados fornecidos, organizados em um dicionário
dados = {
    "Capacidade operacional": [8, 8, 9, 7, 8, 7, 7, 8, 9, 8, 9, 8, 8, 9, 7, 8, 10, 9, 9, 8,
                               10, 8, 7, 9, 10, 8, 9, 8, 8, 10, 8, 8, 8, 9, 9, 10, 8, 8, 10, 9],
    "Adequação às operações": [8, 8, 9, 8, 7, 8, 8, 7, 9, 9, 9, 8, 6, 9, 7, 10, 10, 8, 7, 8,
                               8, 8, 9, 8, 10, 9, 9, 10, 10, 9, 10, 10, 9, 8, 6, 9, 8, 9, 9, 10],
    "Facilidade de operação": [9, 8, 8, 8, 8, 8, 8, 7, 7, 9, 8, 7, 7, 7, 8, 8, 7, 9, 8, 9,
                               9, 7, 8, 9, 8, 10, 9, 10, 10, 10, 10, 9, 8, 8, 7, 10, 9, 9, 9, 10],
    "Conforto e Ergonomia": [9, 8, 9, 8, 8, 8, 10, 9, 9, 7, 10, 8, 9, 8, 10, 9, 10, 8, 8, 8,
                             8, 9, 10, 10, 8, 9, 9, 10, 9, 10, 9, 10, 10, 9, 9, 10, 9, 8, 9, 8],
    "Disponibilidade e Confiabilidade": [8, 8, 7, 5, 7, 7, 6, 7, 7, 9, 8, 7, 7, 6, 7, 9, 10, 8, 7, 6,
                                         6, 8, 8, 7, 7, 8, 6, 7, 8, 8, 8, 9, 8, 8, 5, 8, 10, 10, 1, 9],
    "Facilidade para Manutenções": [8, 8, 6, 6, 8, 7, 7, 7, 8, 8, 9, 7, 8, 7, 8, 2, 8, 7, 8, 10,
                                    7, 9, 9, 8, 8, 9, 8, 8, 8, 4, 7, 7, 7, 9, 10, 8, 7, 10, 10, 10],
    "Custo de Manutenção": [7, 8, 2, 3, 7, 6, 1, 7, 7, 7, 6, 5, 5, 7, 7, 5, 8, 8, 7, 8, 8, 8, 6, 6,
                            2, 8, 7, 8, 8, 9, 6, 10, 8, 10, 8, 7, 2, 7, 8, 7, 10, 7],
    "Consumo de Combustível": [6, 8, 8, 7, 5, 6, 7, 7, 8, 7, 7, 7, 6, 8, 5, 8, 7, 8, 6, 8, 10, 8, 8, 6,
                               8, 8, 8, 5, 8, 7, 7, 7, 9, 9, 8, 7, 8, 9, 9, 10, 8, 7, 7, 7],
    "Adaptabilidade": [8, 9, 8, 8, 6, 7, 8, 7, 8, 8, 9, 7, 9, 8, 7, 9, 10, 9, 7, 8, 8, 9, 8, 9, 5, 7,
                      8, 8, 8, 9, 9, 10, 7, 8, 10, 10, 10, 8, 8, 10, 8, 9, 10, 9, 10, 10],
    "Facilidade de Uso": [9, 8, 8, 8, 7, 7, 8, 7, 6, 9, 7, 7, 4, 7, 5, 10, 6, 8, 10, 10, 7, 9, 9, 9, 8, 8,
                          8, 8, 6, 8, 8, 8, 9, 8, 10, 7, 8, 10, 10, 10, 8, 6, 10, 7, 9, 9, 6, 6, 10, 10],
    "Geração Dados para Gestão da Frota": [8, 8, 0, 8, 5, 7, 6, 7, 8, 7, 9, 3, 5, 6, 5, 8, 10, 10, 8, 8, 8, 7, 8, 9, 10, 6,
                                          0, 10, 10, 9, 9, 10, 8, 10, 10, 10, 9, 4, 10, 10, 10, 5, 10, 8],
    "Geração Dados para Gestão Agrícola": [8, 8, 0, 8, 6, 7, 6, 7, 7, 8, 9, 3, 5, 6, 5, 9, 9, 10, 8, 8, 9, 7, 8, 8,
                                          10, 8, 8, 10, 6, 0, 10, 10, 9, 9, 9, 10, 10, 10, 9, 9, 10, 10, 10, 5, 10]
}

# Para garantir que as listas tenham o mesmo comprimento, vamos transpor as listas
dados_transpostos = list(zip(*dados.values()))

# Convertendo para um array numpy 2D
dados_array = np.array(dados_transpostos)

# Inicializando o modelo KMeans para 3 clusters (por exemplo)
kmeans = KMeans(n_clusters=3, random_state=42)

# Aplicando o KMeans
kmeans.fit(dados_array)

# Obter os rótulos de cada ponto
rotulos = kmeans.labels_

# Plotando o gráfico
plt.figure(figsize=(10, 6))
for i in range(len(rotulos)):
    plt.scatter(i, np.mean(dados_array[i, :]), c=rotulos[i], cmap='viridis', marker='o', s=100)

plt.title('Gráfico KMeans - Agrupamento dos Dados')
plt.xlabel('Índice dos Pontos')
plt.ylabel('Média dos Valores')
plt.colorbar(label='Cluster')
plt.show()
