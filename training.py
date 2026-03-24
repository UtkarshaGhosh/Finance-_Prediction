import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import joblib
import os

# ==========================================
# 1. Configuration & Hyperparameters
# ==========================================
FILE_PATH = 'Data/Stocks/aapl.us.txt' # UPDATE THIS to your actual dataset path
SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.001

# ==========================================
# 2. Data Loading & Preprocessing
# ==========================================
print("Loading and preprocessing data...")
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"Cannot find {FILE_PATH}. Please check the path.")

df = pd.read_csv(FILE_PATH)
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Extract the 'Close' prices
data = df.filter(['Close']).values

# Scale the data to (0, 1) for the LSTM
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Split into train (80%) and test (20%)
training_data_len = int(np.ceil(len(data) * 0.8))
train_data = scaled_data[0:int(training_data_len), :]

# Create training sequences
x_train, y_train = [], []
for i in range(SEQUENCE_LENGTH, len(train_data)):
    x_train.append(train_data[i-SEQUENCE_LENGTH:i, 0])
    y_train.append(train_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Convert to PyTorch tensors and create DataLoader
x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1) 

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# 3. Define the PyTorch LSTM Model
# ==========================================
class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] # Take the output from the last time step
        out = self.fc1(out)
        out = self.fc2(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = LSTMStockPredictor().to(device)

# ==========================================
# 4. Training Loop
# ==========================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

print(f"Starting training on device: {device}...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        # Forward pass
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss/len(train_loader):.6f}')

# ==========================================
# 5. Testing & Evaluation
# ==========================================
print("Evaluating model...")
test_data = scaled_data[training_data_len - SEQUENCE_LENGTH: , :]
x_test, y_test = [], data[training_data_len:, :]

for i in range(SEQUENCE_LENGTH, len(test_data)):
    x_test.append(test_data[i-SEQUENCE_LENGTH:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
x_test_tensor = torch.tensor(x_test, dtype=torch.float32).to(device)

model.eval() 
with torch.no_grad():
    predictions_tensor = model(x_test_tensor)
    
predictions = predictions_tensor.cpu().numpy()
predictions = scaler.inverse_transform(predictions)

rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
print(f'Test RMSE: ${rmse:.2f}')

# ==========================================
# 6. Save the Model and Scaler
# ==========================================
print("Saving model weights and scaler...")

# Save the PyTorch model weights
torch.save(model.state_dict(), 'lstm_stock_model.pth')

# Save the MinMaxScaler (Critical for the web app to unscale future predictions)
joblib.dump(scaler, 'stock_scaler.save')

print("Success! 'lstm_stock_model.pth' and 'stock_scaler.save' are ready for deployment.")

# Optional: Quick plot to verify locally before deploying
train = df[:training_data_len]
valid = df[training_data_len:].copy()
valid['Predictions'] = predictions

plt.figure(figsize=(10,4))
plt.title('Training Complete - Validation Verification')
plt.plot(train['Date'], train['Close'], label='Train')
plt.plot(valid['Date'], valid['Close'], label='Actual')
plt.plot(valid['Date'], valid['Predictions'], label='Predicted')
plt.legend()
plt.show()