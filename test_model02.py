### load packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from scipy.stats import normaltest, zscore
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import json
from sklearn.preprocessing import FunctionTransformer

### load dataset for training
url = 'https://raw.githubusercontent.com/DLBPointon/SummaryStats/version2/test-data/1-1-0-mldata/ML_data-13-08-2024.csv'
df = pd.read_csv(url)


# Missing values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Remove duplicates
df.drop_duplicates(inplace=True)

def remove_outliers_custom(df, column, lower_quantile=0.10, upper_quantile=0.90):
    Q1 = df[column].quantile(lower_quantile)
    Q3 = df[column].quantile(upper_quantile)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

for col in numeric_cols:
    df = remove_outliers_custom(df, col)

df['peak_memory_mb'] = np.log1p(df['peak_memory_mb'])

clade_mapping = dict(enumerate(df['clade'].astype('category').cat.categories))
names_mapping = dict(enumerate(df['names'].astype('category').cat.categories))

with open('clade_mapping.json', 'w') as f:
    json.dump(clade_mapping, f)
with open('names_mapping.json', 'w') as f:
    json.dump(names_mapping, f)

df['clade_numeric'] = df['clade'].map({v: k for k, v in clade_mapping.items()})
df['names_numeric'] = df['names'].map({v: k for k, v in names_mapping.items()})

df = df.drop(columns=['clade', 'names'])

features = ['genome_size', 'cram_total', 'pacbio_total', 'cram_containers', 'clade_numeric', 'names_numeric']
X = df[features]

X = pd.get_dummies(X, columns=['clade_numeric', 'names_numeric'])

training_columns = X.columns

transformer = FunctionTransformer(np.log1p, validate=True)
X_transformed = transformer.fit_transform(X)

y = df['peak_memory_mb']
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.4),
    Dense(1)
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# 5. Treinar o modelo
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32, callbacks=[early_stopping])

loss = model.evaluate(X_test, y_test)
print(f"Loss: {loss}")

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Calculate additional metrics
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs. Predicted Values (Parity Plot)')
plt.show()


model.save('model_peak_memory.h5')

with open('training_columns.json', 'w') as f:
    json.dump(list(training_columns), f)

from tensorflow.keras.models import load_model
model = load_model('model_peak_memory.h5')

with open('training_columns.json', 'r') as f:
    training_columns = json.load(f)

# 2. load new df
url = 'https://raw.githubusercontent.com/DLBPointon/SummaryStats/version2/test-data/1-1-0-mldata/ML_data.csv'
df_new = pd.read_csv(url)

df_new

# Load the saved mappings
with open('clade_mapping.json', 'r') as f:
    clade_mapping = json.load(f)

with open('names_mapping.json', 'r') as f:
    names_mapping = json.load(f)

# Apply the categorical codes to the 'clade' and 'names' columns
df_new['clade_numeric'] = df_new['clade'].map({v: k for k, v in clade_mapping.items()})
df_new['names_numeric'] = df_new['names'].map({v: k for k, v in names_mapping.items()})

# Drop the original categorical columns
df_new = df_new.drop(columns=['clade', 'names'])

# Select the same features
features = ['genome_size', 'cram_total', 'pacbio_total', 'cram_containers', 'clade_numeric', 'names_numeric']
X_new = df_new[features]

# Generate dummy variables for the new dataset
X_new = pd.get_dummies(X_new, columns=['clade_numeric', 'names_numeric'])

# Reindex the new dataset to match the training columns, filling missing columns with 0
X_new = X_new.reindex(columns=training_columns, fill_value=0)

# Apply the same log1p transformation to the features
transformer = FunctionTransformer(np.log1p, validate=True)
X_new_transformed = transformer.fit_transform(X_new)

predictions = model.predict(X_new_transformed)

predicted_peak_memory_mb = predictions
print(predictions)

y_real = df_new['peak_memory_mb']
mae = mean_absolute_error(y_real, predicted_peak_memory_mb)
rmse = np.sqrt(mean_squared_error(y_real, predicted_peak_memory_mb))
r2 = r2_score(y_real, predicted_peak_memory_mb)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")
