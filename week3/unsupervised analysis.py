# KMEANS TEST

features = [
    'peak_memory_mb',
    'genome_size',
    'pacbio_total',
    'cram_total',
    'clade_numeric']

data = df[features].dropna()

# Normalize data
scaler = StandardScaler()
data_normalized = scaler.fit_transform(data)

# elbow method to identify clusters (k)

inertia = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data_normalized)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia, marker='o')
plt.xlabel('N of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow method')
plt.show()
#define clusters as 3
k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(data_normalized)

df['cluster'] = clusters

cluster_means = df.groupby('cluster')[features].mean()
print(cluster_means)

import seaborn as sns

sns.pairplot(df, hue='cluster', vars=features)
plt.show()

df['kmeans_cluster'] = kmeans.labels_
cluster_counts = df['kmeans_cluster'].value_counts()
print(cluster_counts)

clusters_processes_kmeans = df.groupby('kmeans_cluster')['names'].apply(lambda x: x.tolist())

print("Processos em cada cluster (KMeans):")
for cluster_id, processes in clusters_processes_kmeans.items():
    print(f"Cluster {cluster_id}:")
    print(processes)
    print("\n")

# DBSCAN TEST

from sklearn.cluster import DBSCAN
X = df[features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.9, min_samples=5)
clusters = dbscan.fit_predict(X_scaled)
df['dbscan_cluster'] = clusters

sns.pairplot(df, hue='dbscan_cluster', vars=features, palette='Set1')
plt.show()
cluster_counts = df['dbscan_cluster'].value_counts()
print(cluster_counts)

plt.figure(figsize=(10, 6))
plt.scatter(df['peak_memory_mb'], df['genome_size'], c=df['dbscan_cluster'], cmap='tab20', marker='o')
plt.xlabel('Peak Memory MB')
plt.ylabel('Genome Size')
plt.title('DBSCAN Clustering Results (2D)')
plt.colorbar(label='Cluster ID')
plt.show()

eps = 0.5
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
clusters = dbscan.fit_predict(data_normalized)

df_grouped['cluster'] = clusters
cluster_means = df_grouped.groupby('cluster').mean()
print(cluster_means)

df['cluster'] = df['names'].map(df_grouped['cluster'])

sns.pairplot(df, hue='cluster', vars=features)
plt.show()
clusters_processes = df.groupby('cluster')['names'].apply(lambda x: x.tolist())

print("Processos em cada cluster (DBSCAN):")
for cluster_id, processes in clusters_processes.items():
    print(f"Cluster {cluster_id}:")
    print(processes)
    print("\n")

#PCA WITH KMEANS
k = 3  # Defina o número de clusters desejado
kmeans = KMeans(n_clusters=k, random_state=42)
df_grouped['kmeans_cluster'] = kmeans.fit_predict(pca_components)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='kmeans_cluster', palette='viridis', data=df_grouped)
plt.title('PC with KMeans Clustering')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Cluster')
plt.show()

cluster_means = df_grouped.groupby('kmeans_cluster')[features].mean()
print(cluster_means)


#PCA WITH DBSCAN

eps = 0.5
min_samples = 5
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df_grouped['dbscan_cluster'] = dbscan.fit_predict(pca_components)

plt.figure(figsize=(10, 6))
sns.scatterplot(x='PCA1', y='PCA2', hue='dbscan_cluster', palette='viridis', data=df_grouped)
plt.title('PC with DBSCAN Clustering')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.legend(title='Cluster')
plt.show()

dbscan_cluster_means = df_grouped.groupby('dbscan_cluster')[features].mean()
print(dbscan_cluster_means)