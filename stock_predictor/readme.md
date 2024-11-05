# Stock Predictor Pro

A professional stock prediction tool that uses machine learning to analyze stocks and provide buy/sell recommendations with technical indicators.

## Features

- Real-time stock search with auto-suggestions
- Technical analysis with multiple indicators:
  - Moving Averages (MA20, MA50, MA200)
  - Relative Strength Index (RSI)
  - Volume Analysis
  - Candlestick Charts
- Interactive charts with drawing tools
- Dark/Light theme support
- Buy/Sell predictions with confidence scores

## Installation



1. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure
```
stock_predictor/
│
├── data/                   # Data storage
├── model/                  # Model files
├── static/                 # Static files
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
├── templates/              # HTML templates
│   └── index.html
├── app.py                  # Flask application
├── data_collection.py      # Data collection script
├── technical_indicators.py # Technical analysis
└── requirements.txt        # Project dependencies
```

## Training the Model

1. First, collect and process the training data:
```bash
python data_collection.py
```
This will:
- Download historical data for predefined stocks
- Calculate technical indicators
- Save processed data to data/processed_data.pkl

2. Train the model:
```bash
python model/train.py
```
This will:
- Load the processed data
- Train the LSTM model
- Save the trained model to model/stock_predictor.pth

## Running the Application

1. Start the Flask server:
```bash
python app.py
```

2. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage Guide

1. **Search for a Stock**:
   - Use the search bar to find stocks
   - Type stock symbol or company name
   - Select from auto-suggestions

2. **Analyze Stock**:
   - Click "Analyze Stock" button
   - Wait for the prediction and charts to load
   - Review the buy/sell recommendation

3. **Technical Analysis**:
   - Use drawing tools for trend lines
   - Toggle between different indicators
   - Analyze candlestick patterns

4. **Customize View**:
   - Toggle dark/light theme
   - Resize charts as needed
   - Use chart controls for zooming/panning

## Model Details

The prediction model uses:
- LSTM (Long Short-Term Memory) neural network
- Input features:
  - Closing prices
  - Trading volume
  - Technical indicators (RSI, MAs)
- Sequence length: 60 days
- Prediction: Binary (Buy/Sell)

Model architecture:
- Input layer: 6 features
- LSTM layers: 2 (128 units)
- Dropout: 0.2
- Dense layers: 64 units → 2 units
- Output: Softmax probabilities

## Error Handling

Common issues and solutions:

1. **Model file not found**:
   - Ensure you've run the training script
   - Check model/stock_predictor.pth exists

2. **No data available**:
   - Verify internet connection
   - Check if stock symbol is valid
   - Ensure sufficient historical data

3. **Invalid predictions**:
   - Retrain model with updated data
   - Check for data preprocessing issues
   - Verify technical indicator calculations

## Development

To modify or extend the project:

1. **Add New Features**:
   - Add new technical indicators in technical_indicators.py
   - Extend model architecture in model/stock_predictor.py
   - Add new visualization options in static/js/script.js

2. **Improve Model**:
   - Collect more training data
   - Add feature engineering
   - Tune hyperparameters
   - Implement cross-validation

3. **Enhance UI**:
   - Modify templates/index.html for layout changes
   - Update static/css/style.css for styling
   - Add new interactive features in static/js/script.js

## Dependencies

- Python 3.8+
- PyTorch
- Flask
- yfinance
- Plotly
- NumPy
- Pandas
- scikit-learn

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data provided by Yahoo Finance
- Technical analysis implementations based on standard formulas
- UI components inspired by professional trading platforms

## Support

For support, please open an issue in the GitHub repository or contact the maintainers.

## Disclaimer

This tool is for educational purposes only. Always do your own research and consult financial advisors before making investment decisions.
```

This README provides:
1. Complete setup instructions
2. Detailed usage guide
3. Model training process
4. Project structure
5. Development guidelines
6. Error handling
7. Dependencies
8. Contributing guidelines

Users can follow this guide to:
1. Set up the environment
2. Train the model
3. Run the application
4. Understand the codebase
5. Troubleshoot issues
6. Make modifications
```
