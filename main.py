from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import io
import os
from datetime import timedelta

app = FastAPI()

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

class LSTMStockPredictor(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2):
        super(LSTMStockPredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.fc2 = nn.Linear(25, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        out = self.fc1(out)
        out = self.fc2(out)
        return out

device = torch.device('cpu')
model = LSTMStockPredictor().to(device)
model.load_state_dict(torch.load('lstm_stock_model.pth', map_location=device, weights_only=True))
scaler = joblib.load('stock_scaler.save')

@app.get("/")
async def serve_frontend():
    if not os.path.exists("index.html"):
        return {"error": "index.html file not found!"}
    return FileResponse("index.html")

# NEW: Added "period" as a Form parameter
@app.post("/predict")
async def predict_stock(file: UploadFile = File(...), period: str = Form("max")):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    #  NEW: Filter Data based on User Selection
    if period != 'max':
        years = int(period.replace('y', ''))
        cutoff_date = df['Date'].max() - pd.DateOffset(years=years)
        df = df[df['Date'] >= cutoff_date]
        
    data = df.filter(['Close']).values
    if len(data) <= 60:
        return {"error": "Dataset too small for the selected timeframe. Need at least 60 days."}

    scaled_data = scaler.transform(data)
    sequence_length = 60
    
    # --- PART 1: Historical Inference (Testing on known data) ---
    x_test = []
    for i in range(sequence_length, len(scaled_data)):
        x_test.append(scaled_data[i-sequence_length:i, 0])
    x_test = np.reshape(np.array(x_test), (len(x_test), sequence_length, 1))
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    
    model.train() 
    n_iterations = 20 # Reduced slightly for speed with the new future calculation
    mc_predictions = []
    
    with torch.no_grad():
        for _ in range(n_iterations):
            preds = scaler.inverse_transform(model(x_test_tensor).numpy())
            mc_predictions.append(preds.flatten())
            
    mc_predictions = np.array(mc_predictions)
    pred_mean = mc_predictions.mean(axis=0)
    pred_std = mc_predictions.std(axis=0)
    
    upper_bound = pred_mean + (1.96 * pred_std)
    lower_bound = pred_mean - (1.96 * pred_std)
    
    actual_prices = data[sequence_length:].flatten()
    valid_dates = df['Date'].iloc[sequence_length:].dt.strftime('%Y-%m-%d').tolist()
    rmse = float(np.sqrt(np.mean(((pred_mean - actual_prices) ** 2))))

    # --- PART 2: Future Forecasting (Predicting next 365 Days) ---
    future_steps = 365
    future_mc_preds = []
    
    # Get the very last 60 days of known scaled data to start the chain
    last_60_days = scaled_data[-sequence_length:].reshape(1, sequence_length, 1)
    
    with torch.no_grad():
        for _ in range(10):  # 10 MC simulations for the future to keep latency low
            current_seq = last_60_days.copy()
            sim_preds = []
            
            for _ in range(future_steps):
                # Predict next day
                pred = model(torch.tensor(current_seq, dtype=torch.float32)).numpy()[0,0]
                sim_preds.append(pred)
                
                # Append prediction to sequence, remove the oldest day
                new_pred_reshaped = np.array([[[pred]]])
                current_seq = np.concatenate((current_seq[:, 1:, :], new_pred_reshaped), axis=1)
                
            future_mc_preds.append(sim_preds)

    future_mc_preds = np.array(future_mc_preds)
    # Unscale future predictions
    future_mc_preds_unscaled = np.array([scaler.inverse_transform(run.reshape(-1, 1)).flatten() for run in future_mc_preds])
    
    future_mean = future_mc_preds_unscaled.mean(axis=0)
    future_std = future_mc_preds_unscaled.std(axis=0)
    future_upper = future_mean + (1.96 * future_std)
    future_lower = future_mean - (1.96 * future_std)
    
    # Generate future dates
    last_date = df['Date'].iloc[-1]
    future_dates = [(last_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(1, future_steps + 1)]

    # --- AI Summary Logic ---
    trend_percent = ((future_mean[-1] - actual_prices[-1]) / actual_prices[-1]) * 100
    risk_percent = (np.mean(future_upper - future_lower) / np.mean(future_mean)) * 100

    if risk_percent > 35: ai_summary = f"⚠️ <b>High Volatility Warning:</b> 1-Year future epistemic uncertainty is massive (Risk Factor: {risk_percent:.1f}%). Recommendation: <b>HOLD</b>."
    elif trend_percent > 2.0: ai_summary = f"🚀 <b>Bullish Signal:</b> 1-Year forecast projects a <b>{trend_percent:.1f}% increase</b>. Recommendation: <b>ACCUMULATE/BUY.</b>"
    elif trend_percent < -2.0: ai_summary = f"📉 <b>Bearish Signal:</b> 1-Year forecast projects a <b>{abs(trend_percent):.1f}% decline</b>. Recommendation: <b>SELL/HEDGE.</b>"
    else: ai_summary = f"⚖️ <b>Neutral Outlook:</b> 1-Year forecast predicts sideways movement ({trend_percent:.1f}%). Recommendation: <b>HOLD.</b>"

    return {
        "dates": valid_dates, "actual": actual_prices.tolist(),
        "predicted": pred_mean.tolist(), "upper_bound": upper_bound.tolist(), "lower_bound": lower_bound.tolist(),
        "future_dates": future_dates, "future_mean": future_mean.tolist(),
        "future_upper": future_upper.tolist(), "future_lower": future_lower.tolist(),
        "rmse": rmse, "ai_summary": ai_summary
    }