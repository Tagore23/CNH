import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

# Dados fornecidos
data = {
    "Capacidade Operacional": [8, 8, 9, 7, 8, 7, 7, 8, 9, 8, 9, 8, 8, 9, 7, 8, 10, 9, 9, 8, 10, 8, 7, 9, 10, 8, 9, 8, 8, 10, 8, 8, 8, 9, 9, 10, 10, 10, 10, 9, 9, 10, 8, 9, 8, 9, 9, 9, 10, 9],
    "Adequação": [8, 8, 9, 8, 7, 8, 8, 7, 9, 9, 9, 8, 6, 9, 7, 10, 10, 8, 7, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 10, 9, 9, 6, 8, 8, 9, 9, 10, 9, 9, 9, 10, 9],
    "Facilidade de Operação": [9, 8, 8, 8, 8, 8, 8, 7, 7, 9, 8, 7, 7, 7, 8, 8, 7, 8, 8, 8, 8, 7, 7, 8, 8, 9, 8, 7, 10, 10, 8, 8, 8, 9, 9, 9, 9, 8, 9, 8, 8, 8, 9, 9, 9, 8, 9, 9, 9, 9],
    "Conforto e Ergonomia": [9, 8, 9, 8, 8, 8, 10, 9, 9, 7, 10, 8, 9, 8, 10, 9, 8, 8, 9, 8, 9, 10, 10, 10, 10, 10, 9, 9, 8, 8, 9, 9, 8, 8, 9, 10, 9, 10, 9, 8, 9, 8, 9, 8, 9, 8, 9, 10, 9, 9],
    "Disponibilidade e Confiabilidade": [8, 8, 7, 5, 7, 7, 6, 7, 7, 9, 8, 7, 7, 6, 7, 9, 10, 8, 7, 6, 6, 8, 8, 7, 7, 8, 8, 6, 6, 4, 7, 8, 8, 7, 8, 7, 7, 9, 10, 8, 8, 5, 8, 10, 10, 1, 9, 7, 8, 8],
    "Facilidade para Manutenção": [8, 8, 6, 6, 8, 7, 7, 7, 8, 8, 9, 7, 8, 7, 8, 2, 8, 7, 8, 10, 7, 9, 9, 8, 8, 9, 8, 8, 8, 4, 7, 7, 7, 9, 10, 8, 7, 10, 10, 10, 8, 8, 2, 7, 8, 9, 10, 8, 8, 9],
    "Custo de Manutenção": [7, 8, 2, 3, 7, 6, 1, 7, 7, 7, 6, 5, 7, 7, 5, 8, 8, 7, 8, 8, 8, 6, 6, 2, 8, 7, 8, 8, 8, 7, 7, 8, 7, 6, 9, 7, 6, 8, 8, 8, 7, 7, 8, 9, 6, 10, 8, 10, 8, 7],
    "Consumo de Combustível": [6, 8, 8, 7, 5, 6, 7, 7, 8, 7, 7, 7, 6, 8, 5, 8, 7, 8, 6, 8, 10, 8, 8, 6, 8, 8, 8, 5, 8, 7, 7, 7, 9, 9, 8, 7, 8, 9, 9, 10, 8, 7, 7, 7, 10, 7, 7, 8, 9, 10],
    "Adaptabilidade": [8, 9, 8, 8, 6, 7, 8, 7, 8, 8, 9, 7, 9, 8, 7, 9, 10, 9, 7, 8, 8, 9, 8, 9, 9, 8, 8, 9, 5, 7, 8, 8, 8, 9, 9, 10, 7, 8, 10, 10, 10, 8, 8, 10, 8, 9, 10, 9, 10, 9],
    "Facilidade de Uso": [9, 8, 8, 8, 7, 7, 8, 7, 6, 9, 7, 7, 4, 7, 5, 10, 6, 8, 10, 10, 7, 9, 9, 9, 8, 8, 8, 8, 6, 8, 8, 8, 9, 8, 10, 7, 8, 10, 10, 10, 8, 6, 10, 7, 9, 9, 6, 6, 10, 10],
    "Gestão de Dados de Frota": [8, 8, 0, 8, 5, 7, 6, 7, 8, 7, 9, 3, 5, 6, 5, 8, 10, 10, 8, 8, 8, 7, 8, 9, 10, 8, 9, 10, 6, 0, 10, 10, 9, 9, 10, 8, 10, 10, 10, 9, 4, 10, 10, 10, 5, 10, 8, 9, 10, 8],
    "Gestão de Dados Agrícolas": [8, 8, 0, 8, 6, 7, 6, 7, 7, 8, 9, 3, 5, 6, 5, 9, 9, 10, 8, 8, 9, 7, 8, 8, 10, 8, 8, 10, 6, 0, 10, 10, 9, 9, 9, 10, 10, 10, 9, 9, 10, 10, 10, 5, 10, 8, 8, 9, 10, 10]
}

# Encontrando o comprimento máximo
max_length = max(len(lst) for lst in data.values())

# Preenchendo listas mais curtas com NaN
for key in data:
    while len(data[key]) < max_length:
        data[key].append(np.nan)

# Agora podemos criar uma matriz com as listas
all_data = np.array([data[key] for key in data]).T

# Verificando se as listas têm o mesmo comprimento
list_lengths = {key: len(value) for key, value in data.items()}
assert len(set(list_lengths.values())) == 1, "As listas têm comprimentos diferentes!"

# Plotando o gráfico de dispersão
plt.figure(figsize=(10, 6))
plt.scatter(all_data[:, 0], all_data[:, 1], c=all_data[:, 2], cmap='viridis')
plt.title("Gráfico de Dispersão")
plt.xlabel("Capacidade Operacional")
plt.ylabel("Adequação")
plt.colorbar(label="Facilidade de Operação")
plt.show()

# Aplicando o modelo GMM
gmm = GaussianMixture(n_components=3, covariance_type='full')
gmm.fit(all_data)
predictions = gmm.predict(all_data)

# Plotando os resultados do GMM
plt.figure(figsize=(10, 6))
plt.scatter(all_data[:, 0], all_data[:, 1], c=predictions, cmap='viridis')
plt.title("Clusters Identificados pelo GMM")
plt.xlabel("Capacidade Operacional")
plt.ylabel("Adequação")
plt.colorbar(label="Cluster")
plt.show()
