import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# Importing CSV file.
df = pd.read_csv('houses_to_rent.csv')

dropped_columns = [df.columns[0]]
df.drop(dropped_columns, axis = 1, inplace=True)

encoder = OneHotEncoder(sparse_output=False)
df['furniture'] = encoder.fit_transform(df[['furniture']]).astype('int64')
df['animal'] = encoder.fit_transform(df[['animal']]).astype('int64')

object_to_int_columns = np.append(np.array(df.columns[list(df.columns).index('hoa') : list(df.columns).index('total')+1]), 'floor')

def non_numeric_column(column):
    df[column] = df[column].str.replace(r'[^0-9]', '', regex=True).str.strip()

for i in range(len(object_to_int_columns)):
    non_numeric_column(object_to_int_columns[i])

def handle_empty_strings(column):
    df[column] = df[column].replace('', '0')

# Handling empty strings
for col in object_to_int_columns:
    handle_empty_strings(col)

def object_to_int(column):
    df[column] = df[column].astype('float')

for index in range(len(object_to_int_columns)):
    object_to_int(object_to_int_columns[index])

# Calculate Z-score for numeric columns
numeric_columns = df.select_dtypes(include=np.number).columns
z_scores = zscore(df[numeric_columns])

# Define threshold for outlier detection
threshold = 3

# Find outliers
outlier_indices = np.where(np.abs(z_scores) > threshold)

"""# Print the indices of outliers
print("Indices of outliers:")
for col, idx in zip(numeric_columns[outlier_indices[1]], outlier_indices[0]):
    print(f"Column: {col}, Index: {idx}")"""

# Calculate median for each numeric column
column_medians = df[numeric_columns].median()

# Replace outliers with column medians
for col, idx in zip(numeric_columns[outlier_indices[1]], outlier_indices[0]):
    df.at[idx, col] = column_medians[col]

"""# Verify that outliers have been replaced
print("Outliers replaced with column medians:")
for col, idx in zip(numeric_columns[outlier_indices[1]], outlier_indices[0]):
    print(f"Column: {col}, Index: {idx}, New value: {df.at[idx, col]}")"""

X = df.drop('total', axis=1).values
y = df['total'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = (X_train - X_train.mean()) / X_train.std()
X_test = (X_test - X_test.mean()) / X_test.std()

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

class TheDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
train_dataset = TheDataset(X_train_tensor, y_train_tensor)
test_dataset = TheDataset(X_test_tensor, y_test_tensor)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(NeuralNet, self).__init__()
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
# Define the sizes of the network layers
input_size = X_train.shape[1]
hidden_size1 = 64
hidden_size2 = 32
output_size = 1 

model = NeuralNet(input_size, hidden_size1, hidden_size2, output_size)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

num_epochs = 30
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

print('Training finished')

model.eval()
test_loss = 0.0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        test_loss += loss.item() * inputs.size(0)

avg_test_loss = test_loss / len(test_loader.dataset)
print(f'Average Test Loss: {avg_test_loss:.4f}')

# Generate random data
num_samples = 10  # Number of random samples
num_features = X_train.shape[1]  # Number of features
random_data = np.random.randn(num_samples, num_features)  # Generating random data with the same shape as X_train

# Preprocess the random data (normalize, etc.)
random_data_normalized = (random_data - X_train.mean()) / X_train.std()
random_data_tensor = torch.tensor(random_data_normalized, dtype=torch.float32)

# Make predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    predictions = model(random_data_tensor)

# Convert predictions tensor to numpy array
predictions_array = predictions.numpy()

# Print the predictions
print("Predictions for random data:")
print(predictions_array)
