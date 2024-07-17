!pip install jovian --upgrade --quiet

from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import jovian

df = 'https://raw.githubusercontent.com/DLBPointon/SummaryStats/version2/test-data/1-1-0-mldata/ML_data.csv'

urlretrieve(df, 'ML_data.csv')
df = pd.read_csv('ML_data.csv')
df

df.info()

df['status'].value_counts()

df['clade'].value_counts()

df.describe()

print(df.isnull().sum())

df['clade'].fillna(df['clade'].mode()[0], inplace=True)
print(df.isnull().sum())

sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'

fig = px.scatter(df,
                 x='genome_size',
                 y='peak_memory_mb',
                 color='cpu_used',
                 opacity=0.8,
                 hover_data=['clade'],
                 title='')
fig.update_traces(marker_size=5)
fig.show()

df.cpus_requested.corr(df.memory_requested_mb)

df.genome_size.corr(df.peak_memory_mb)

df.genome_size.corr(df.cpu_used)

df.genome_size.corr(df.average_memory_used_as_percentage)

unique_categories = df['clade'].unique()

category_to_numeric = {category: idx for idx, category in enumerate(unique_categories)}

df['clade_numeric'] = df['clade'].map(category_to_numeric)

print("\nDataFrame com 'clade_numeric':")
print(df[['clade', 'clade_numeric']])

unique_categories = df['status'].unique()

category_to_numeric = {category: idx for idx, category in enumerate(unique_categories)}

df['status_numeric'] = df['status'].map(category_to_numeric)

print("\nDataFrame com 'clade_numeric':")
print(df[['status', 'status_numeric']])

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
for col in numeric_columns:
    fig = px.histogram(df, x=col, marginal='box', nbins=50, title=f'Distribution {col}')
    fig.update_layout(bargap=0.1)
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted values')
plt.show()

print(df[['cram_total', 'cram_file_no', 'cram_containers']].describe())

# CRAM variables distribution
sns.histplot(df['cram_total'], kde=True)
plt.title('Distribuição de CRAM Total')
plt.xlabel('CRAM Total')
plt.ylabel('Frequência')
plt.show()

sns.histplot(df['cram_file_no'], kde=True)
plt.title('Distribuição de Número de Arquivos CRAM')
plt.xlabel('Número de Arquivos CRAM')
plt.ylabel('Frequência')
plt.show()

sns.histplot(df['cram_containers'], kde=True)
plt.title('Distribuição de Containers CRAM')
plt.xlabel('Containers CRAM')
plt.ylabel('Frequência')
plt.show()

correlations = df[['memory_requested_mb', 'cram_total', 'cram_file_no', 'cram_containers']].corr()
print(correlations)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


X = df[['cram_total']]
y = df['memory_requested_mb']

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)


rmse = mean_squared_error(y, y_pred, squared=False)
r2 = r2_score(y, y_pred)

print('Regressão Linear Simples com CRAM Total')
print('Coeficientes:', model.coef_)
print('Intercept:', model.intercept_)
print('RMSE:', rmse)
print('R^2:', r2)

selected_columns = ['genome_size', 'pacbio_total', 'pacbio_file_no', 'cram_total', 'cram_file_no', 'cram_containers', 'clade_numeric']
data_selected = df[selected_columns]

def plot_correlation_matrix(data, title):
    correlation_matrix = data.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(title)
    plt.show()
data_selected = df[selected_columns]

plot_correlation_matrix(data_selected, 'Correlation matrix of causative variables')


print(df.columns)

df['unique_name']

df['pacbio_file_no']

unique_names = df['unique_name'].unique()
for name in unique_names:
    df_filtered = df[df['unique_name'] == name][features]
    corr_matrix_filtered = df_filtered.corr()
    plt.figure(figsize=(6, 6))
    sns.heatmap(corr_matrix_filtered, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Correlation matrix - Subset for {name}')
    plt.show()

from sklearn.cluster import KMeans
import numpy as np


correlation_matrices = []
for name in unique_names:
    df_filtered = df[df['unique_name'] == name][features]
    corr_matrix_filtered = df_filtered.corr()
    correlation_matrices.append(corr_matrix_filtered.values.flatten())

correlation_matrices = np.array(correlation_matrices)

# KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(correlation_matrices)
labels = kmeans.labels_

# Add labels
summary_df['cluster'] = labels

plt.figure(figsize=(12, 6))
sns.scatterplot(x='unique_name', y='avg_correlation', hue='cluster', data=summary_df, palette='viridis')
plt.xticks(rotation=90)
plt.title('Correlation Clusters between Causative Variables')
plt.tight_layout()
plt.savefig('clusters_correlation.png')
plt.show()

for cluster in range(3):
    print(f'\nSubsets on Cluster {cluster}:')
    print(summary_df[summary_df['cluster'] == cluster]['unique_name'].values)


plt.figure(figsize=(12, 6))
scatter = plt.scatter(x='unique_name', y='avg_correlation', data=summary_df, c='avg_correlation', s=100, cmap='coolwarm', alpha=0.7)
plt.xticks(rotation=90, fontsize=8)
plt.title('Average Correlation between Causative Variables by Subset')
plt.colorbar(scatter, label='Correlação Média')
plt.tight_layout()
plt.savefig('scatter_corr_mean.png')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='avg_correlation', data=summary_df)
plt.title('Distribution of Average Correlations between Causative Variables')
plt.tight_layout()
plt.savefig('boxplot_corr_mean.png')
plt.show()

for cluster in range(3):
    cluster_data = summary_df[summary_df['cluster'] == cluster]
    avg_correlation_matrix = np.mean([df[df['unique_name'] == name][features].corr().values.flatten() for name in cluster_data['unique_name']], axis=0)
    avg_correlation_matrix_df = pd.DataFrame(avg_correlation_matrix.reshape(len(features), len(features)), index=features, columns=features)

    plt.figure(figsize=(10, 8))
    sns.heatmap(avg_correlation_matrix_df, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(f'Average Correlation Matrix for the Cluster {cluster}')
    plt.tight_layout()
    plt.savefig(f'corr_matrix_cluster{cluster}.png')
    plt.show()
