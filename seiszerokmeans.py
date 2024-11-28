import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer

# Todos os dados fornecidos
dados = {
    'Capacidade operacional (hectares por hora) (CSAT)': [6, 6, 5, 4, 6, 6, 8, 5, 3, 5, 5, 6, 5, 1, 5, 8, 7, 0, 8, 5, 8, 3, 10, 9, 0, 3, 7, 10, 5, 10, 6, 8, 5, 5],
    'Adequação às Diversas Operações e Implementos (CSAT)': [6, 5, 5, 3, 6, 5, 8, 8, 2, 6, 7, 5, 4, 7, 5, 8, 7, 0, 7, 5, 8, 5, 10, 8, 0, 3, 7, 10, 5, 10, 6, 8, 5, 8],
    'Facilidade de Operação (CSAT)': [6, 7, 3, 6, 6, 6, 10, 8, 3, 6, 8, 5, 3, 8, 5, 8, 7, 0, 9, 5, 8, 7, 7, 7, 0, 3, 7, 7, 5, 4, 8, 8, 8, 10],
    'Conforto e Ergonomia (CSAT)': [6, 7, 3, 3, 6, 6, 10, 8, 3, 2, 8, 6, 6, 1, 8, 8, 8, 0, 9, 5],
    'Disponibilidade e Confiabilidade Mecânica (CSAT)': [1, 5, 3, 6, 5, 1, 8, 5, 3, 1, 3, 4, 2, 0, 5, 6, 6, 0, 5, 4, 0, 0, 5, 7, 0, 0, 2, 7, 0, 4, 6, 1, 2, 0],
    'Facilidade para Realização de Manutenções (CSAT)': [1, 7, 3, 6, 7, 6, 8, 6, 2, 5, 7, 5, 5, 8, 5, 8, 7, 0, 5, 8, 8, 5, 4, 3, 0, 0, 6, 7, 5, 9, 8, 8, 6, 8],
    'Custo de Manutenção (CSAT)': [1, 5, 2, 6, 5, 2, 6, 3, 4, 5, 4, 5, 0, 4, 3, 0, 4, 2, 7, 5, 3, 5, 0, 1, 5, 5, 0, 5, 8, 1, 9, 5],
    'Consumo de Combustível (litros por hectare) (CSAT)': [6, 5, 3, 5, 4, 2, 10, 3, 2, 3, 7, 4, 5, 5, 4, 6, 3, 0, 9, 6, 7, 3, 7, 7, 0, 4, 6, 6, 5, 9, 8, 7, 5, 8],
    'Adaptabilidade às Mais Diversas Condições de Trabalho (CSAT)': [6, 5, 5, 7, 7, 6, 10, 8, 3, 3, 7, 4, 5, 8, 0, 8, 6, 0, 9, 6, 8, 5, 9, 8, 0, 2, 7, 0, 9, 8, 8, 5, 8],
    'Facilidade de Uso do Piloto Automático (CSAT)': [8, 7, 2, 7, 7, 7, 10, 8, 1, 6, 8, 5, 2, 1, 2, 8, 8, 0, 8, 0, 8, 5, 4, 8, 0, 0, 7, 8, 5, 9, 6, 6, 7, 10],
    'Geração e Transmissão de Dados para Gestão da Frota (CSAT)': [0, 7, 3, 6, 6, 6, 10, 2, 1, 8, 6, 2, 0, 8, 8, 8, 5, 4, 0, 0, 8, 8, 0, 6, 7, 8, 9],
    'Geração e Transmissão de Dados para Gestão Agrícola (CSAT)': [2, 8, 4, 4, 6, 6, 10, 2, 1, 8, 6, 5, 0, 8, 8, 8, 5, 4, 5, 8, 8, 6, 7, 3, 9],
}

# Ajustar para tamanhos iguais preenchendo com NaN
tamanho_maximo = max(len(valores) for valores in dados.values())
for categoria in dados:
    dados[categoria] += [np.nan] * (tamanho_maximo - len(dados[categoria]))

# Converter para matriz NumPy
valores_array = np.array([dados[categoria] for categoria in dados]).T

# Imputar NaNs usando a média
imputer = SimpleImputer(strategy='mean')
valores_array = imputer.fit_transform(valores_array)

# Aplicar KMeans
kmeans = KMeans(n_clusters=3, random_state=0, n_init='auto')
kmeans.fit(valores_array)
labels = kmeans.labels_

# Plotar os clusters usando duas primeiras categorias
plt.figure(figsize=(12, 8))
plt.scatter(valores_array[:, 0], valores_array[:, 1], c=labels, cmap='viridis', marker='o')
plt.title("K-Means Clustering com Todos os Dados")
plt.xlabel("Capacidade operacional (hectares por hora) (CSAT)")
plt.ylabel("Adequação às Diversas Operações (CSAT)")
plt.show()
