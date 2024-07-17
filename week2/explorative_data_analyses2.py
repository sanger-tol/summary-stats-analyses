!pip install jovian --upgrade --quiet
from urllib.request import urlretrieve
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import jovian
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
import numpy as np
from sklearn.preprocessing import StandardScaler

df = 'https://raw.githubusercontent.com/DLBPointon/SummaryStats/version2/test-data/1-1-0-mldata/ML_data.csv'


urlretrieve(df, 'ML_data.csv')

df = pd.read_csv('ML_data.csv')

df.head()

# Inspect the unique values in the 'names' column
unique_names = df['names'].unique()

# Print the unique names to understand the processes
print(unique_names)


print(df.head())
print(df.info())
print(df.describe())

missing_values = df.isnull().sum()
print(missing_values)


numeric_cols = df.select_dtypes(include=[float, int]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

df['cpus_requested'] = df['cpus_requested'].astype(float)
df['memory_requested_mb'] = df['memory_requested_mb'].astype(float)

df['clade_numeric'] = df['clade'].astype('category').cat.codes

features_of_interest = [
    'realtime_seconds',
    'average_memory_used_as_percentage',
    'peak_memory_mb',
    'cpu_used',
    'genome_size',
    'pacbio_total',
    'cram_total',
    'cram_containers',
    'clade_numeric'
]

df[features_of_interest] = StandardScaler().fit_transform(df[features_of_interest])

df = pd.get_dummies(df, columns=['names'], drop_first=True)

unique_processes = df.filter(like='names_').columns  


process_dataframes = {}
for process in unique_processes:
    process_dataframes[process] = df[df[process] == 1]


def plot_correlation_matrix(data, process_name):
    data_of_interest = data[features_of_interest]
    correlation_matrix = data_of_interest.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    plt.title(f'Correlation Matrix for process: {process_name}')
    plt.show()

for process, data in process_dataframes.items():
    plot_correlation_matrix(data, process)



df_selected = df[features_of_interest]

correlation_matrix = df_selected.corr()

correlation_df = correlation_matrix.unstack().reset_index()
correlation_df.columns = ['Feature1', 'Feature2', 'Correlation']

correlation_df = correlation_df[correlation_df['Feature1'] != correlation_df['Feature2']]


correlation_df['pair'] = correlation_df.apply(lambda row: tuple(sorted([row['Feature1'], row['Feature2']])), axis=1)
correlation_df = correlation_df.drop_duplicates(subset='pair')
correlation_df = correlation_df.drop(columns='pair')


def categorize_correlation(value):
    if abs(value) > 0.7:
        return 'Highly correlated'
    elif abs(value) > 0.5:
        return 'Moderately correlated'
    else:
        return 'Low correlation'

correlation_df['Category'] = correlation_df['Correlation'].apply(categorize_correlation)


category_counts = correlation_df['Category'].value_counts()


plt.figure(figsize=(10, 6))
category_counts.plot(kind='bar')
plt.title('Distribution of Correlation Categories')
plt.xlabel('Correlation Category')
plt.ylabel('Number of Feature Pairs')
plt.xticks(rotation=45)
plt.show()

for category in correlation_df['Category'].unique():
    subset = correlation_df[correlation_df['Category'] == category]
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Correlation', y='Feature1', hue='Feature2', data=subset, dodge=False)
    plt.title(f'Correlation Pairs in Category: {category}')
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature Pairs')
    plt.show()




features_of_interest = [
    'realtime_seconds',
    'average_memory_used_as_percentage',
    'genome_size',
    'pacbio_total',
    'cram_total',
    'cram_containers',
    'clade_numeric'
]


df_selected = df[features_of_interest + ['peak_memory_mb', 'cpu_used']]


correlation_matrix = df_selected.corr()

correlations_memory = correlation_matrix['peak_memory_mb'].drop(['peak_memory_mb', 'cpu_used']).sort_values(ascending=False)

correlations_cpu = correlation_matrix['cpu_used'].drop(['peak_memory_mb', 'cpu_used']).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=correlations_memory.values, y=correlations_memory.index)
plt.title('Correlation with Peak Memory Usage (peak_memory_mb)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x=correlations_cpu.values, y=correlations_cpu.index)
plt.title('Correlation with CPU Usage (cpu_used)')
plt.xlabel('Correlation Coefficient')
plt.ylabel('Features')
plt.show()
