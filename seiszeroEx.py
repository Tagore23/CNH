import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.impute import SimpleImputer

# Dados fornecidos (os mesmos de antes)
data = {
    "Capacidade Operacional (hectares por hora)": [6, 6, 5, 4, 6, 6, 8, 5, 3, 5, 5, 6, 5, 1, 5, 8, 7, 0, 8, 5, 8, 3, 10, 9, 0, 3, 7, 10, 5, 10, 6, 8, 5, 5],
    "Adequação às Diversas Operações e Implementos": [6, 5, 5, 3, 6, 5, 8, 8, 2, 6, 7, 5, 4, 7, 5, 8, 7, 0, 7, 5, 8, 5, 10, 8, 0, 3, 7, 10, 5, 10, 6, 8, 5, 8],
    "Facilidade de Operação": [6, 7, 3, 6, 6, 6, 10, 8, 3, 6, 8, 5, 3, 8, 5, 8, 7, 0, 9, 5, 8, 7, 7, 7, 0, 3, 7, 7, 5, 4, 8, 8, 8, 10],
    "Conforto e Ergonomia": [6, 7, 3, 3, 6, 6, 10, 8, 3, 2, 8, 6, 6, 1, 8, 8, 8, 0, 9, 5],
    "Disponibilidade e Confiabilidade Mecânica": [1, 5, 3, 6, 5, 1, 8, 5, 3, 1, 3, 4, 2, 0, 5, 6, 6, 0, 5, 4, 0, 0, 5, 7, 0, 0, 2, 7, 0, 4, 6, 1, 2, 0],
    "Facilidade para Realização de Manutenções": [1, 7, 3, 6, 7, 6, 8, 6, 2, 5, 7, 5, 5, 8, 5, 8, 7, 0, 5, 8, 8, 5, 4, 3, 0, 0, 6, 7, 5, 9, 8, 8, 6, 8],
    "Custo de Manutenção": [1, 5, 2, 6, 5, 2, 6, 3, 4, 5, 4, 5, 0, 4, 3, 0, 4, 2, 7, 5, 3, 5, 0, 1, 5, 5, 0, 5, 8, 1, 9, 5],
    "Consumo de Combustível (litros por hectare)": [6, 5, 3, 5, 4, 2, 10, 3, 2, 3, 7, 4, 5, 5, 4, 6, 3, 0, 9, 6, 7, 3, 7, 7, 0, 4, 6, 6, 5, 9, 8, 7, 5, 8],
    "Adaptabilidade às Mais Diversas Condições de Trabalho": [6, 5, 5, 7, 7, 6, 10, 8, 3, 3, 7, 4, 5, 8, 0, 8, 6, 0, 9, 6, 8, 5, 9, 8, 0, 2, 7, 0, 9, 8, 8, 5],
    "Facilidade de Uso do Piloto Automático": [8, 7, 2, 7, 7, 7, 10, 8, 1, 6, 8, 5, 2, 1, 2, 8, 8, 0, 8, 0, 8, 5, 4, 8, 0, 0, 7, 8, 5, 9, 6, 6, 7, 10],
    "Geração e Transmissão de Dados para Gestão da Frota": [0, 7, 3, 6, 6, 6, 10, 2, 1, 8, 6, 2, 0, 8, 8, 8, 5, 4, 0, 0, 8, 8, 0, 6, 7, 8, 9],
    "Geração e Transmissão de Dados para Gestão Agrícola": [2, 8, 4, 4, 6, 6, 10, 2, 1, 8, 6, 5, 0, 8, 8, 8, 5, 4, 5, 8, 8, 6, 7, 3, 9]
}

# Encontrando o comprimento máximo
max_length = max(len(lst) for lst in data.values())

# Preenchendo listas mais curtas com NaN
for key in data:
    while len(data[key]) < max_length:
        data[key].append(np.nan)

# Criando a matriz de dados
all_data = np.array([data[key] for key in data]).T

# Usando o SimpleImputer para preencher NaN com a média
imputer = SimpleImputer(strategy='mean')
all_data_imputed = imputer.fit_transform(all_data)

# Plotando o gráfico de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(all_data_imputed[:, 0], all_data_imputed[:, 1], c=all_data_imputed[:, 2], cmap='viridis')
plt.title("Gráfico de Dispersão")
plt.xlabel("Capacidade Operacional (hectares por hora)")
plt.ylabel("Adequação às Diversas Operações e Implementos")
plt.colorbar(label="Facilidade de Operação")
plt.show()

# Aplicando o modelo GMM
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(all_data_imputed)
predictions = gmm.predict(all_data_imputed)

# Plotando os resultados do GMM
plt.figure(figsize=(10, 6))
plt.scatter(all_data_imputed[:, 0], all_data_imputed[:, 1], c=predictions, cmap='viridis')
plt.title("Clusters Identificados pelo GMM")
plt.xlabel("Capacidade Operacional (hectares por hora)")
plt.ylabel("Adequação às Diversas Operações e Implementos")
plt.colorbar(label="Cluster")
plt.show()
