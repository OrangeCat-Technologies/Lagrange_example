from flask import Flask, render_template, request, jsonify
import yfinance as yf
import torch
import torch.nn.functional as F
import numpy as np
from model.stock_predictor import StockPredictor
from technical_indicators import add_technical_indicators, calculate_rsi
import pickle
import os
import plotly.graph_objs as go
import plotly.utils
import json
import traceback
import requests

app = Flask(__name__)

print("Starting Flask application...")

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)
print("Model directory checked/created")

try:
    print("Loading model...")
    model = StockPredictor()
    if os.path.exists('model/stock_predictor.pth'):
        model.load_state_dict(torch.load('model/stock_predictor.pth', 
                                        map_location=torch.device('cpu'),
                                        weights_only=True))
        print("Model loaded successfully")
    else:
        print("WARNING: Model file not found!")
    model.eval()

except Exception as e:
    print(f"Error during initialization: {str(e)}")
    print(traceback.format_exc())
    raise e

def validate_ticker(symbol):
    try:
        ticker = yf.Ticker(symbol)
        # Get historical data to verify if the ticker is valid and has data
        hist = ticker.history(period='1mo')
        if len(hist) > 0:
            # Additional check for valid price data
            if hist['Close'].iloc[-1] > 0:
                return True
        return False
    except:
        return False

@app.route('/')
def home():
    print("\nAccessing home page")
    return render_template('index.html')

@app.route('/search_ticker', methods=['POST'])
def search_ticker():
    try:
        query = request.json.get('query', '').upper()
        if not query:
            return jsonify([])

        # Search for tickers using Yahoo Finance API
        url = f"https://query2.finance.yahoo.com/v1/finance/search?q={query}&quotesCount=10&newsCount=0"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        data = response.json()

        suggestions = []
        for quote in data.get('quotes', []):
            if quote.get('symbol') and quote.get('shortname'):
                # Filter out non-stock symbols (like cryptocurrencies, mutual funds)
                if not any(x in quote['symbol'] for x in ['-', '^', '=']):
                    suggestions.append({
                        'symbol': quote['symbol'],
                        'name': quote['shortname']
                    })

        return jsonify(suggestions)
    except Exception as e:
        print(f"Error in search_ticker: {str(e)}")
        return jsonify([])

@app.route('/validate_ticker', methods=['POST'])
def check_ticker():
    try:
        symbol = request.json.get('symbol', '').upper()
        if not symbol:
            return jsonify({'valid': False, 'error': 'No symbol provided'})

        # Basic format validation
        if not symbol.isalnum():
            return jsonify({'valid': False, 'error': 'Invalid symbol format'})

        is_valid = validate_ticker(symbol)
        return jsonify({'valid': is_valid})
    except Exception as e:
        print(f"Error validating ticker: {str(e)}")
        return jsonify({'valid': False, 'error': str(e)})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        symbol = request.form['symbol'].upper()
        print(f"\nReceived prediction request for {symbol}")
        
        # Validate ticker before proceeding
        if not validate_ticker(symbol):
            print(f"Invalid symbol: {symbol}")
            return jsonify({'error': 'Invalid symbol or no data available for this stock'})
        
        input_data, hist_data = prepare_prediction_data(symbol)
        print(f"Input data shape: {input_data.shape}")
        
        if torch.isnan(input_data).any():
            print("WARNING: NaN values found in input data")
            input_data = torch.nan_to_num(input_data, nan=0.0)
        
        with torch.no_grad():
            output = model(input_data)
            print(f"Model output: {output}")
            
            if torch.isnan(output).any():
                print("WARNING: NaN values in model output")
                output = torch.nan_to_num(output, nan=0.0)
            
            prediction = torch.argmax(output, dim=1).item()
            print(f"Prediction index: {prediction}")
            confidence = calculate_confidence(output)
            print(f"Confidence: {confidence}")
        
        result = "BUY" if prediction == 1 else "SELL"
        print(f"Final prediction: {result}")
        
        confidence_str = f"{confidence*100:.2f}%"
        graphs = create_stock_graphs(symbol, hist_data)
        
        probabilities = F.softmax(torch.nan_to_num(output, nan=0.0), dim=1)[0].tolist()
        sell_prob = f"{probabilities[0]*100:.2f}%"
        buy_prob = f"{probabilities[1]*100:.2f}%"
        print(f"Sell probability: {sell_prob}")
        print(f"Buy probability: {buy_prob}")
        
        prediction_details = {
            'symbol': symbol,
            'prediction': result,
            'confidence': confidence_str,
            'raw_confidence': confidence,
            'graphs': graphs,
            'sell_probability': sell_prob,
            'buy_probability': buy_prob
        }
        
        return jsonify(prediction_details)
    
    except Exception as e:
        print(f"Error in predict route: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': f"Prediction error: {str(e)}"})

def calculate_confidence(output):
    try:
        print(f"Calculating confidence for output shape: {output.shape}")
        if torch.isnan(output).any():
            print("WARNING: NaN values found in model output")
            output = torch.nan_to_num(output, nan=0.0)
        
        probabilities = F.softmax(output, dim=1)
        print(f"Probabilities: {probabilities}")
        
        confidence = torch.max(probabilities).item()
        print(f"Calculated confidence: {confidence}")
        return confidence
    except Exception as e:
        print(f"Error in calculate_confidence: {str(e)}")
        print(traceback.format_exc())
        return 0.0

def prepare_prediction_data(symbol):
    try:
        print(f"\nPreparing prediction data for {symbol}")
        stock = yf.Ticker(symbol)
        hist = stock.history(period='3mo')
        print(f"Downloaded historical data: {len(hist)} rows")
        
        if len(hist) < 60:
            print(f"Insufficient data: only {len(hist)} rows")
            raise ValueError("Insufficient historical data")
        
        hist = add_technical_indicators(hist)
        
        features = ['Close', 'Volume', 'RSI', 'MA20', 'MA50', 'MA200']
        data = hist[features].values[-60:]
        
        print("Feature statistics before normalization:")
        for i, feature in enumerate(features):
            print(f"{feature}: mean={np.mean(data[:, i]):.2f}, std={np.std(data[:, i]):.2f}")
        
        means = np.nanmean(data, axis=0)
        stds = np.nanstd(data, axis=0)
        stds[stds == 0] = 1
        
        normalized_data = (data - means) / stds
        
        print("\nFeature statistics after normalization:")
        for i, feature in enumerate(features):
            print(f"{feature}: mean={np.mean(normalized_data[:, i]):.2f}, std={np.std(normalized_data[:, i]):.2f}")
        
        if np.isnan(normalized_data).any():
            print("WARNING: NaN values found in normalized data")
            normalized_data = np.nan_to_num(normalized_data, nan=0.0)
        
        return torch.FloatTensor(normalized_data).unsqueeze(0), hist
    
    except Exception as e:
        print(f"Error in prepare_prediction_data: {str(e)}")
        print(traceback.format_exc())
        raise Exception(f"Error preparing data for {symbol}: {str(e)}")

def create_stock_graphs(symbol, hist_data):
    try:
        print(f"\nCreating graphs for {symbol}")
        
        # Create candlestick chart
        candlestick = go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name='Candlestick'
        )
        
        # Add Moving Averages
        ma_traces = []
        for ma in ['MA20', 'MA50']:
            ma_traces.append(go.Scatter(
                x=hist_data.index,
                y=hist_data[ma],
                name=f'{ma}',
                line=dict(dash='dash')
            ))
        
        price_layout = go.Layout(
            title=f'{symbol} Stock Price History',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            hovermode='x unified'
        )
        
        price_fig = go.Figure(data=[candlestick] + ma_traces, layout=price_layout)
        
        # Create volume chart
        volume_colors = ['red' if row['Open'] > row['Close'] else 'green' 
                        for _, row in hist_data.iterrows()]
        volume_trace = go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name='Volume',
            marker=dict(color=volume_colors)
        )
        
        volume_layout = go.Layout(
            title=f'{symbol} Trading Volume',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Volume')
        )
        
        volume_fig = go.Figure(data=[volume_trace], layout=volume_layout)
        
        # Create RSI chart
        rsi_trace = go.Scatter(
            x=hist_data.index,
            y=hist_data['RSI'],
            name='RSI'
        )
        
        rsi_layout = go.Layout(
            title=f'{symbol} RSI',
            xaxis=dict(title='Date'),
            yaxis=dict(title='RSI')
        )
        
        rsi_fig = go.Figure(data=[rsi_trace], layout=rsi_layout)
        
        print("Graphs created successfully")
        
        return {
            'price': json.loads(json.dumps(price_fig, cls=plotly.utils.PlotlyJSONEncoder)),
            'volume': json.loads(json.dumps(volume_fig, cls=plotly.utils.PlotlyJSONEncoder)),
            'rsi': json.loads(json.dumps(rsi_fig, cls=plotly.utils.PlotlyJSONEncoder))
        }
    
    except Exception as e:
        print(f"Error in create_stock_graphs: {str(e)}")
        print(traceback.format_exc())
        raise e

if __name__ == '__main__':
    print("\nStarting Flask development server...")
    app.run(debug=True)