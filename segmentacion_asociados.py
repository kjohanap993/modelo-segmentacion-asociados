# =========================================================
# PROYECTO DE GRADO – ANALÍTICA DE DATOS
# Modelo de segmentación de asociados mediante K-means
# =========================================================

# -------------------------
# 1. LIBRERÍAS
# -------------------------
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# -------------------------
# 2. SEMILLA (REPRODUCIBILIDAD)
# -------------------------
np.random.seed(42)
random.seed(42)

# -------------------------
# 3. GENERACIÓN DEL DATASET SINTÉTICO
# -------------------------
categorias = ['Basica', 'Intermedia', 'Plena']
prob_categorias = [0.21, 0.39, 0.40]

generos = ['Masculino', 'Femenino']
prob_generos = [0.45, 0.55]

niveles_estudios = ['Sin educacion', 'Primaria', 'Secundaria', 'Tecnico', 'Universitario']

ocupaciones = ['Empleado', 'Independiente', 'Pensionado', 'Estudiante', 'Ama de casa']
prob_ocupaciones = [0.3, 0.3, 0.15, 0.1, 0.15]

has_credit = ['Si', 'No']
prob_credit = [0.5, 0.5]

zonas = ['Urbano', 'Rural']
prob_zona_por_cat = {
    'Basica': [0.3, 0.7],
    'Intermedia': [0.7, 0.3],
    'Plena': [0.9, 0.1]
}

n = 1000
data = []

for _ in range(n):
    cat = random.choices(categorias, prob_categorias)[0]

    edad = int(np.random.normal(45, 15))
    edad = max(18, min(90, edad))

    gen = random.choices(generos, prob_generos)[0]

    ingresos = abs(int(np.random.normal(3000000, 2000000)))

    if cat == 'Basica':
        prob_niv = [0.2, 0.3, 0.3, 0.15, 0.05]
    elif cat == 'Intermedia':
        prob_niv = [0.1, 0.2, 0.3, 0.25, 0.15]
    else:
        prob_niv = [0.05, 0.15, 0.25, 0.25, 0.3]

    nivel = random.choices(niveles_estudios, prob_niv)[0]

    frec_trans = max(0, int(np.random.normal(10, 5)))

    ocup = random.choices(ocupaciones, prob_ocupaciones)[0]
    credit = random.choices(has_credit, prob_credit)[0]

    depositos = abs(int(np.random.normal(ingresos * 1.5, ingresos)))

    zona = random.choices(zonas, prob_zona_por_cat[cat])[0]

    # Recencia simulada (días desde última transacción)
    recencia = max(1, int(np.random.exponential(scale=30)))

    data.append([
        cat, edad, gen, ingresos, nivel,
        frec_trans, ocup, credit, depositos, zona, recencia
    ])

df = pd.DataFrame(
    data,
    columns=[
        'Categoria', 'Edad', 'Genero', 'Ingresos',
        'Nivel_Estudios', 'Frec_Trans_Mes',
        'Ocupacion', 'Has_Credit', 'Depositos',
        'Zona', 'Recencia'
    ]
)

# -------------------------
# 4. CÁLCULO DE INDICADORES RFM
# -------------------------
df['R'] = df['Recencia']
df['F'] = df['Frec_Trans_Mes']
df['M'] = df['Depositos']

# Escoring por quintiles
df['R_score'] = pd.qcut(df['R'], 5, labels=[5, 4, 3, 2, 1])
df['F_score'] = pd.qcut(df['F'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
df['M_score'] = pd.qcut(df['M'], 5, labels=[1, 2, 3, 4, 5])

df[['R_score', 'F_score', 'M_score']] = df[['R_score', 'F_score', 'M_score']].astype(int)

# -------------------------
# 5. SELECCIÓN Y ESCALADO DE VARIABLES
# -------------------------
variables_modelo = df[['Edad', 'Ingresos', 'F', 'M', 'R_score', 'F_score', 'M_score']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(variables_modelo)

# -------------------------
# 6. MÉTODO DEL CODO
# -------------------------
inertia = []

k_range = range(2, 8)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

plt.figure()
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Número de clústeres (k)')
plt.ylabel('Inercia')
plt.title('Método del Codo')
plt.show()

# -------------------------
# 7. MODELO FINAL K-MEANS (k = 4)
# -------------------------
k_final = 4
kmeans_final = KMeans(n_clusters=k_final, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(X_scaled)

# -------------------------
# 8. VALIDACIÓN – SILHOUETTE
# -------------------------
silhouette_avg = silhouette_score(X_scaled, df['Cluster'])
print(f"Silhouette Score (k={k_final}): {silhouette_avg:.3f}")

# -------------------------
# 9. ANÁLISIS DESCRIPTIVO POR CLÚSTER
# -------------------------
resumen_clusters = df.groupby('Cluster')[[
    'Edad', 'Ingresos', 'F', 'M', 'R'
]].mean()

print("\nResumen estadístico por clúster:")
print(resumen_clusters)

# -------------------------
# 10. EXPORTACIÓN DE RESULTADOS
# -------------------------
df.to_csv('dataset_segmentado.csv', index=False)
resumen_clusters.to_csv('resumen_clusters.csv')

print("\nArchivos generados:")
print("- dataset_segmentado.csv")
print("- resumen_clusters.csv")
