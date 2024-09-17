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

df['clade_numeric'] = df['clade'].astype('category').cat.codes
df['names_numeric'] = df['names'].astype('category').cat.codes
df = df.drop(columns=['clade', 'names'])
print(df.head())

from sklearn.preprocessing import FunctionTransformer

# Apply log1p transformation to the target variable
df['peak_memory_mb'] = np.log1p(df['peak_memory_mb'])


# Select specific features
features = ['genome_size', 'cram_total', 'pacbio_total', 'cram_containers', 'clade_numeric', 'names_numeric']
X = df[features]
X = pd.get_dummies(X, columns=['clade_numeric', 'names_numeric'])
# Target variable
y = df['peak_memory_mb']

# Apply log1p transformation to all features
transformer = FunctionTransformer(np.log1p, validate=True)
X_transformed = transformer.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(X_transformed, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

training_columns = pd.get_dummies(df[['clade_numeric', 'names_numeric']]).columns

from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(64, activation='relu', kernel_regularizer=l2(0.001)),
    Dropout(0.2),
    Dense(1)
])

# Compile the model with a lower learning rate
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mean_squared_error')

# Train the model with early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_val, y_val),
    callbacks=[early_stopping]
)

# Evaluate the model
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

model.save('modelo_peak_memory.h5')