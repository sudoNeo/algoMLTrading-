import pandas as pd
import numpy as np
import yfinance as yf
import pandas_ta as ta
import matplotlib.pyplot as plt
import xgboost as xgb
import alpaca_trade_api as tradeapi
import json
import os
import logging
import time
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from pykalman import KalmanFilter
from tenacity import retry, wait_exponential, stop_after_attempt

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trading_system.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration settings for the trading system."""
    def __init__(self, config_file='config.json'):
        # Default configuration
        self.tickers = ['AAPL', 'BA']
        self.start_date = '2014-01-01'
        self.end_date = datetime.now().strftime('%Y-%m-%d')
        self.alpaca_api_key = 'YOUR_API_KEY'  # Replace with your actual API key
        self.alpaca_secret_key = 'YOUR_SECRET_KEY'  # Replace with your actual secret key
        self.alpaca_base_url = 'https://paper-api.alpaca.markets'
        self.alpaca_data_url = 'https://data.alpaca.markets'
        self.initial_capital = 10000
        self.risk_per_trade = 0.01  # 1% risk per trade
        self.max_daily_loss = 0.015  # 1.5% max daily loss
        self.max_weekly_loss = 0.05  # 5% max weekly loss
        self.max_sector_exposure = 0.15  # 15% max sector exposure
        self.retraining_frequency = 14  # Retrain every 14 days
        self.sharpe_threshold = 1.0  # Retrain if Sharpe drops below this
        self.consecutive_loss_days = 3  # Retrain after this many loss days
        self.selected_strategy = 'combined'  # Options: 'mean_reversion', 'momentum', 'stat_arb', 'combined'
        self.use_alpaca_api = True  # Set to False to disable Alpaca API usage if keys are not set
        
        # Load from config file if exists
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config_data = json.load(f)
                    for key, value in config_data.items():
                        setattr(self, key, value)
                logger.info(f"Loaded configuration from {config_file}")
            except Exception as e:
                logger.error(f"Error loading config file: {e}")
                
        # Check if API keys are set properly
        if self.alpaca_api_key == 'YOUR_API_KEY' or self.alpaca_secret_key == 'YOUR_SECRET_KEY':
            logger.warning("Alpaca API keys are not configured. Live trading will be disabled.")
            self.use_alpaca_api = False
    
    def save(self, config_file='config.json'):
        """Save current configuration to file."""
        with open(config_file, 'w') as f:
            json.dump(self.__dict__, f, indent=4)


class DataHandler:
    """Handles data acquisition and preprocessing."""
    def __init__(self, config):
        self.config = config
        self.tickers = config.tickers
        self.start_date = config.start_date
        self.end_date = config.end_date
        self.data = {}
        
    def fetch_historical_data(self):
        """Fetch historical data from yfinance."""
        logger.info(f"Fetching historical data for {self.tickers}")
        
        # First try to fetch all tickers at once with group_by='ticker'
        try:
            all_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                progress=False,
                group_by='ticker'  # Group data by ticker
            )
            
            # For multiple tickers, extract each ticker's data
            for ticker in self.tickers:
                if ticker in all_data.columns.levels[0]:
                    # Get ticker data (this will have proper OHLCV columns)
                    ticker_data = all_data[ticker].copy()
                    self.data[ticker] = ticker_data
                    logger.info(f"Downloaded {len(ticker_data)} rows for {ticker}")
        except Exception as e:
            logger.error(f"Error in bulk download: {e}")
            # Fall back to individual downloads
            for ticker in self.tickers:
                try:
                    # Add auto_adjust=False to ensure we get standard columns
                    data = yf.download(
                        ticker,
                        start=self.start_date,
                        end=self.end_date,
                        progress=False,
                        auto_adjust=False
                    )
                    
                    # Handle unusual column structure
                    if all(col == ticker for col in data.columns):
                        logger.warning(f"Received column headers with only ticker names for {ticker}")
                        # Create proper price columns manually
                        price_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
                        if len(data.columns) >= len(price_columns):
                            renamed_data = pd.DataFrame(index=data.index)
                            for i, col_name in enumerate(price_columns):
                                if i < len(data.columns):
                                    renamed_data[col_name] = data.iloc[:, i]
                            self.data[ticker] = renamed_data
                            logger.info(f"Fixed columns for {ticker}: now {renamed_data.columns.tolist()}")
                        else:
                            logger.error(f"Couldn't fix columns for {ticker}: expected at least {len(price_columns)} columns but got {len(data.columns)}")
                    else:
                        self.data[ticker] = data
                    
                    logger.info(f"Downloaded {len(data)} rows for {ticker}")
                except Exception as e2:
                    logger.error(f"Error downloading data for {ticker}: {e2}")
        
        return self.data
    
    def fetch_fundamental_data(self):
        """Fetch fundamental data from yfinance."""
        logger.info("Fetching fundamental data")
        fundamental_data = {}
        for ticker in self.tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                fundamental_data[ticker] = {
                    'PE_Ratio': info.get('trailingPE', None),
                    'PB_Ratio': info.get('priceToBook', None),
                    'Dividend_Yield': info.get('dividendYield', None),
                    'Debt_to_Equity': info.get('debtToEquity', None),
                    'ROE': info.get('returnOnEquity', None),
                    'Market_Cap': info.get('marketCap', None)
                }
                logger.info(f"Fetched fundamental data for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching fundamental data for {ticker}: {e}")
        return fundamental_data
    
    def setup_alpaca_api(self):
        """Set up Alpaca API for live trading."""
        # Check if API usage is enabled
        if not self.config.use_alpaca_api:
            logger.warning("Alpaca API usage is disabled in configuration. Skipping connection.")
            return None
            
        # Check if API keys are properly set
        if self.config.alpaca_api_key == 'YOUR_API_KEY' or self.config.alpaca_secret_key == 'YOUR_SECRET_KEY':
            logger.error("Alpaca API keys are not configured. Please update config.json with your API keys.")
            return None
            
        try:
            api = tradeapi.REST(
                self.config.alpaca_api_key,
                self.config.alpaca_secret_key,
                base_url=self.config.alpaca_base_url
            )
            account = api.get_account()
            logger.info(f"Alpaca API connected. Account status: {account.status}")
            return api
        except Exception as e:
            logger.error(f"Error connecting to Alpaca API: {e}")
            logger.info("You can continue with backtest mode, but live trading will be disabled.")
            return None
        
    def fetch_live_data(self, api, timeframe='1Min'):
        """Fetch live data from Alpaca API."""
        live_data = {}
        for ticker in self.tickers:
            try:
                # Get the current time in EST
                now = datetime.now()
                start = now - timedelta(days=5)  # Get 5 days of minute data
                
                # Format timestamps
                start_str = start.strftime('%Y-%m-%dT%H:%M:%SZ')
                now_str = now.strftime('%Y-%m-%dT%H:%M:%SZ')
                
                # Get the data
                barset = api.get_bars(
                    ticker,
                    timeframe,
                    start=start_str,
                    end=now_str
                ).df
                
                live_data[ticker] = barset
                logger.info(f"Fetched {len(barset)} live bars for {ticker}")
            except Exception as e:
                logger.error(f"Error fetching live data for {ticker}: {e}")
        return live_data


class FeatureEngineering:
    """Handles feature engineering and selection."""
    def __init__(self, data_handler):
        self.data_handler = data_handler
        self.features = {}
        
    def calculate_technical_indicators(self):
        """Calculate technical indicators for each ticker."""
        logger.info("Calculating technical indicators")
        for ticker, df in self.data_handler.data.items():
            if df is None or df.empty:
                logger.error(f"No data available for {ticker}")
                continue
            
            # Create a copy of the dataframe to avoid modifying original
            try:
                # Ensure we have the necessary columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(col in df.columns for col in required_cols):
                    logger.error(f"Missing required columns for {ticker}. Available columns: {df.columns.tolist()}")
                    continue
                
                # Create features DataFrame
                self.features[ticker] = pd.DataFrame(index=df.index)
                
                # Copy price data
                for col in required_cols:
                    self.features[ticker][col] = df[col]
                
                # Calculate indicators
                logger.info(f"Calculating indicators for {ticker} with {len(df)} rows")
                
                # Moving Averages
                self.features[ticker]['MA50'] = ta.sma(df['Close'], length=50)
                self.features[ticker]['MA200'] = ta.sma(df['Close'], length=200)
                
                # RSI
                self.features[ticker]['RSI'] = ta.rsi(df['Close'], length=14)
                
                # MACD
                macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
                if macd is not None:
                    for col in macd.columns:
                        self.features[ticker][col] = macd[col]
                
                # Bollinger Bands
                bbands = ta.bbands(df['Close'], length=20)
                if bbands is not None:
                    for col in bbands.columns:
                        self.features[ticker][col] = bbands[col]
                
                # On-Balance Volume
                self.features[ticker]['OBV'] = ta.obv(df['Close'], df['Volume'])
                
                # Add price movement
                self.features[ticker]['Return_1d'] = df['Close'].pct_change(1)
                self.features[ticker]['Return_5d'] = df['Close'].pct_change(5)
                
                # Volatility
                self.features[ticker]['Volatility_20d'] = df['Close'].pct_change().rolling(20).std()
                
                # Kalman Filter smoothed price
                kf = KalmanFilter(initial_state_mean=df['Close'].iloc[0], n_dim_obs=1)
                state_means, _ = kf.filter(df['Close'].values)
                self.features[ticker]['Kalman_Price'] = state_means
                
                # Calculate signal
                self.features[ticker]['price_above_ma50'] = df['Close'] > self.features[ticker]['MA50']
                self.features[ticker]['ma50_above_ma200'] = self.features[ticker]['MA50'] > self.features[ticker]['MA200']
                self.features[ticker]['rsi_oversold'] = self.features[ticker]['RSI'] < 30
                self.features[ticker]['rsi_overbought'] = self.features[ticker]['RSI'] > 70
                
                # Target column - prediction target
                self.features[ticker]['Target'] = df['Close'].shift(-5) > df['Close']
                
                logger.info(f"Successfully calculated indicators for {ticker} with {len(self.features[ticker].columns)} features")
                print(f"Features created for {ticker}: {self.features[ticker].columns.tolist()}")
                
            except Exception as e:
                logger.error(f"Error calculating indicators for {ticker}: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        return self.features
    
    def add_fundamental_features(self, fundamental_data):
        """Add fundamental data to features."""
        for ticker, fund_data in fundamental_data.items():
            if ticker in self.features:
                for key, value in fund_data.items():
                    if value is not None:
                        self.features[ticker][key] = value
                logger.info(f"Added fundamental features for {ticker}")
        return self.features
    
    def select_features(self, correlation_threshold=0.75):
        """Select features based on correlation analysis."""
        selected_features = {}
        
        for ticker, df in self.features.items():
            # Remove NaN values
            clean_df = df.dropna()
            
            # Get only numeric features
            numeric_features = clean_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            
            # Calculate correlation matrix
            corr_matrix = clean_df[numeric_features].corr().abs()
            
            # Select features with correlation below threshold
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > correlation_threshold)]
            
            # Keep important features regardless of correlation
            important_features = ['Close', 'MA50', 'MA200', 'RSI', 'OBV', 'MACD_12_26_9', 'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0', 'Kalman_Price', 'Target', 'Volatility_20d']
            to_drop = [col for col in to_drop if col not in important_features]
            
            selected_features[ticker] = clean_df.drop(to_drop, axis=1)
            logger.info(f"Selected {len(selected_features[ticker].columns)} features for {ticker} after correlation analysis")
        
        return selected_features


class ModelTrainer:
    """Handles model training and prediction."""
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.feature_importance = {}
        self.last_training_date = datetime.now()
        
    def prepare_training_data(self, features_df):
        """Prepare data for model training."""
        try:
            # Check if the dataframe is empty or None
            if features_df is None or features_df.empty:
                logger.error("Empty features dataframe provided for training")
                return None, None, None, None
            
            # Check if we're dealing with a MultiIndex
            if isinstance(features_df.columns, pd.MultiIndex):
                logger.warning("Converting MultiIndex DataFrame to simple index for training")
                features_df.columns = features_df.columns.get_level_values(-1)  # Use last level
                
            # Check if Target column exists
            if 'Target' not in features_df.columns:
                logger.error("Target column missing from features dataframe")
                logger.error(f"Available columns: {features_df.columns.tolist()}")
                return None, None, None, None
                
            # Convert Target to numeric explicitly to avoid potential boolean/object issues
            try:
                features_df['Target'] = features_df['Target'].astype(float).astype(int)
            except Exception as e:
                logger.error(f"Error converting Target to numeric: {e}")
                return None, None, None, None
                
            # Remove NaN values
            df = features_df.dropna()
            
            # Check if we have enough data after dropping NaNs
            if len(df) < 10:  # Lower threshold to handle limited data
                logger.error(f"Insufficient data after dropping NaNs: {len(df)} rows")
                return None, None, None, None
                
            # Define target and features
            y = df['Target']
            
            # Remove non-numeric and future-leaking columns
            numeric_cols = df.select_dtypes(include=['float64', 'int64', 'bool']).columns
            X = df[numeric_cols].copy()
            if 'Target' in X.columns:
                X = X.drop(['Target'], axis=1)
            
            # Check if we have any features left
            if X.empty or len(X.columns) == 0:
                logger.error("No numeric features available for training")
                return None, None, None, None
                
            # Fill any remaining NaN values with 0
            X = X.fillna(0)
                
            # Handle class imbalance with SMOTE
            try:
                # Only use SMOTE if we have enough samples of each class
                y_counts = y.value_counts()
                if len(y_counts) >= 2 and all(y_counts >= 5):
                    smote = SMOTE(random_state=42)
                    X_resampled, y_resampled = smote.fit_resample(X, y)
                else:
                    logger.warning("Not enough samples for SMOTE, using original data")
                    X_resampled, y_resampled = X, y
            except Exception as e:
                logger.error(f"Error applying SMOTE: {e}")
                # If SMOTE fails, just use the original data
                X_resampled, y_resampled = X, y
            
            return X, y, X_resampled, y_resampled
            
        except Exception as e:
            logger.error(f"Unexpected error in prepare_training_data: {e}")
            return None, None, None, None
    
    def train_model(self, ticker, features_df):
        """Train model for a ticker."""
        logger.info(f"Training model for {ticker}")
        
        X, y, X_resampled, y_resampled = self.prepare_training_data(features_df)
        
        # Check if data preparation was successful
        if X is None or y is None or X_resampled is None or y_resampled is None:
            logger.error(f"Data preparation failed for {ticker}. Cannot train model.")
            # Create a dummy model to avoid errors when trying to use the model later
            dummy_model = xgb.XGBClassifier(random_state=42)
            if hasattr(features_df, 'select_dtypes'):
                try:
                    dummy_X = features_df.select_dtypes(include=['float64', 'int64']).iloc[:10]
                    dummy_y = np.random.randint(0, 2, size=len(dummy_X))
                    dummy_model.fit(dummy_X, dummy_y)
                    self.models[ticker] = dummy_model
                    logger.warning(f"Created dummy model for {ticker} to avoid errors")
                except Exception as e:
                    logger.error(f"Could not create dummy model for {ticker}: {e}")
            return None, 0.0
        
        # Use time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Train GBM (XGBoost)
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.01,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='auc',
            use_label_encoder=False,
            random_state=42
        )
        
        try:
            # Train on resampled data
            model.fit(X_resampled, y_resampled)
            
            # Evaluate on original data
            y_pred = model.predict(X)
            accuracy = accuracy_score(y, y_pred)
            
            # Store model and feature importance
            self.models[ticker] = model
            self.feature_importance[ticker] = pd.Series(
                model.feature_importances_, 
                index=X.columns
            ).sort_values(ascending=False)
            
            logger.info(f"Model for {ticker} trained with accuracy: {accuracy:.4f}")
            return model, accuracy
            
        except Exception as e:
            logger.error(f"Error training model for {ticker}: {e}")
            return None, 0.0
    
    def train_alternative_models(self, ticker, features_df):
        """Train alternative models for comparison."""
        X, y, X_resampled, y_resampled = self.prepare_training_data(features_df)
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_resampled, y_resampled)
        rf_pred = rf_model.predict(X)
        rf_accuracy = accuracy_score(y, rf_pred)
        
        # Train SVM
        svm_model = SVC(probability=True, random_state=42)
        svm_model.fit(X_resampled, y_resampled)
        svm_pred = svm_model.predict(X)
        svm_accuracy = accuracy_score(y, svm_pred)
        
        logger.info(f"Alternative models for {ticker} - RF accuracy: {rf_accuracy:.4f}, SVM accuracy: {svm_accuracy:.4f}")
        return {
            'random_forest': (rf_model, rf_accuracy),
            'svm': (svm_model, svm_accuracy)
        }
    
    def predict(self, ticker, features_df, confidence_threshold=0.6):
        """Make predictions using trained model."""
        if ticker not in self.models:
            logger.error(f"No trained model found for {ticker}")
            return None
        
        # Remove NaN values and get features
        df = features_df.dropna()
        X = df.select_dtypes(include=['float64', 'int64'])
        X = X.drop(['Target'], axis=1, errors='ignore')
        
        # Make prediction with probability
        probabilities = self.models[ticker].predict_proba(X)
        # Class 1 probability (probability of price increase)
        buy_probability = probabilities[:, 1]
        
        # Generate signals based on probability threshold
        signals = np.zeros(len(buy_probability))
        signals[buy_probability >= confidence_threshold] = 1  # Buy
        signals[buy_probability <= (1 - confidence_threshold)] = -1  # Sell
        
        return pd.Series(signals, index=df.index)
    
    def should_retrain(self, performance_metrics):
        """Determine if models should be retrained based on criteria."""
        # Check time-based criterion
        days_since_training = (datetime.now() - self.last_training_date).days
        if days_since_training >= self.config.retraining_frequency:
            logger.info(f"Retraining triggered: {days_since_training} days since last training")
            return True
        
        # Check performance-based criteria
        if performance_metrics:
            # Check Sharpe ratio
            if 'sharpe_ratio' in performance_metrics and performance_metrics['sharpe_ratio'] < self.config.sharpe_threshold:
                logger.info(f"Retraining triggered: Sharpe ratio {performance_metrics['sharpe_ratio']} below threshold")
                return True
            
            # Check consecutive loss days
            if 'consecutive_loss_days' in performance_metrics and performance_metrics['consecutive_loss_days'] >= self.config.consecutive_loss_days:
                logger.info(f"Retraining triggered: {performance_metrics['consecutive_loss_days']} consecutive loss days")
                return True
        
        return False


class Strategy:
    """Base class for trading strategies."""
    def __init__(self, config, data_handler):
        self.config = config
        self.data_handler = data_handler
        self.positions = {}
        
    def generate_signals(self, features):
        """Generate trading signals based on strategy."""
        raise NotImplementedError("Subclasses must implement generate_signals method")
    
    def get_positions(self):
        """Get current positions."""
        return self.positions


class MeanReversionStrategy(Strategy):
    """Mean reversion strategy using Kalman filters."""
    def generate_signals(self, features):
        signals = {}
        for ticker, df in features.items():
            try:
                clean_df = df.dropna()
                if clean_df.empty:
                    logger.warning(f"No valid data for {ticker} after dropping NaNs")
                    signals[ticker] = pd.Series(0, index=df.index)
                    continue
                # Safety check for None or empty DataFrame
                if df is None or df.empty:
                    logger.error(f"Empty DataFrame for {ticker}")
                    signals[ticker] = pd.Series(0, index=[])
                    continue
                    
                # Check for required columns before doing anything else
                if 'Close' not in df.columns or 'Kalman_Price' not in df.columns:
                    logger.error(f"Required columns not found for {ticker}")
                    signals[ticker] = pd.Series(0, index=df.index)
                    continue
                
                # Try to copy essential data instead of full DataFrame
                try:
                    # Instead of dropna() on the full DataFrame, just extract what we need
                    closes = df['Close'].copy()
                    kalman = df['Kalman_Price'].copy()
                    
                    # Create a DataFrame with just these columns
                    clean_data = pd.DataFrame({'Close': closes, 'Kalman_Price': kalman})
                    
                    # Now dropna only on these columns
                    clean_data = clean_data.dropna()
                    
                    # Calculate z-score of price deviation from Kalman mean
                    deviation = clean_data['Close'] - clean_data['Kalman_Price']
                    deviation_mean = deviation.rolling(20).mean()
                    deviation_std = deviation.rolling(20).std().replace(0, np.nan).fillna(deviation.std())
                    
                    # Avoid division by zero
                    if deviation_std.min() <= 0:
                        logger.warning(f"Standard deviation too small for {ticker}, using unscaled signals")
                        z_score = deviation - deviation_mean  # Just the deviation without scaling
                    else:
                        z_score = (deviation - deviation_mean) / deviation_std
                    
                    # Generate signals - use vectorized operations
                    result = pd.Series(0, index=clean_data.index)
                    result[z_score < -2.0] = 1  # Buy when price significantly below mean
                    result[z_score > 2.0] = -1  # Sell when price significantly above mean
                    
                    signals[ticker] = result
                    logger.info(f"Generated mean reversion signals for {ticker}")
                    
                except Exception as e:
                    logger.error(f"Error processing {ticker} in mean reversion: {e}")
                    signals[ticker] = pd.Series(0, index=df.index)
                
            except Exception as e:
                logger.error(f"Unexpected error for {ticker} in mean reversion: {e}")
                signals[ticker] = pd.Series(0, index=df.index if df is not None and hasattr(df, 'index') else [])
        
        return signals


class MomentumStrategy(Strategy):
    """Momentum strategy with volatility scaling."""
    def generate_signals(self, features):
        signals = {}
        for ticker, df in features.items():
            clean_df = df.dropna()
            
            # Check for required columns
            required_cols = ['MA50', 'MA200', 'RSI', 'Volatility_20d', 'Close']
            missing_cols = [col for col in required_cols if col not in clean_df.columns]
            
            if missing_cols:
                logger.error(f"Required columns missing for {ticker}: {missing_cols}")
                # Create default empty signals
                signals[ticker] = pd.Series(0, index=clean_df.index)
                continue
                
            try:
                # Generate base signals
                base_signals = pd.Series(0, index=clean_df.index)
                
                # Bullish: MA50 > MA200 and RSI > 50 but not overbought
                bullish = (clean_df['MA50'] > clean_df['MA200']) & (clean_df['RSI'] > 50) & (clean_df['RSI'] < 70)
                
                # Bearish: MA50 < MA200 and RSI < 50 but not oversold
                bearish = (clean_df['MA50'] < clean_df['MA200']) & (clean_df['RSI'] < 50) & (clean_df['RSI'] > 30)
                
                base_signals[bullish] = 1
                base_signals[bearish] = -1
                
                # Check if volatility data is valid before using it
                if clean_df['Volatility_20d'].isna().all() or (clean_df['Volatility_20d'] == 0).all():
                    logger.warning(f"All volatility values are NaN or zero for {ticker}, using unscaled signals")
                    signals[ticker] = base_signals
                else:
                    # Scale signals by inverse volatility - with safety checks
                    volatility = clean_df['Volatility_20d'].replace(0, np.nan)
                    vol_mean = volatility.mean(skipna=True)
                    
                    if pd.isna(vol_mean) or vol_mean == 0:
                        logger.warning(f"Cannot calculate valid volatility mean for {ticker}, using unscaled signals")
                        signals[ticker] = base_signals
                    else:
                        # Fill NaN and 0 values with the mean
                        volatility = volatility.fillna(vol_mean)
                        volatility = volatility.replace(0, vol_mean)
                        
                        # Scale signals
                        vol_scale = 1 / volatility
                        vol_scale = vol_scale / vol_scale.mean()  # Normalize
                        
                        # Apply scaling and thresholds - with array operations instead of apply
                        scaled = base_signals * vol_scale
                        result = pd.Series(0, index=scaled.index)
                        result[scaled > 0.5] = 1
                        result[scaled < -0.5] = -1
                        signals[ticker] = result
                
                logger.info(f"Generated momentum signals for {ticker}")
                
            except Exception as e:
                logger.error(f"Error generating momentum signals for {ticker}: {e}")
                signals[ticker] = pd.Series(0, index=clean_df.index)
        
        return signals


class StatisticalArbitrageStrategy(Strategy):
    """Statistical arbitrage strategy for pairs trading."""
    def generate_signals(self, features):
        if len(self.data_handler.tickers) < 2:
            logger.warning("Statistical arbitrage requires at least 2 tickers")
            return {}
        
        # For simplicity, we'll use the first two tickers as a pair
        pair = self.data_handler.tickers[:2]
        ticker1, ticker2 = pair
        
        if ticker1 not in features or ticker2 not in features:
            logger.error(f"Required data not found for pair {pair}")
            return {}
        
        # Get clean data
        df1 = features[ticker1].dropna()
        df2 = features[ticker2].dropna()
        
        # Ensure same index
        common_idx = df1.index.intersection(df2.index)
        df1 = df1.loc[common_idx]
        df2 = df2.loc[common_idx]
        
        if len(df1) < 60:  # Need enough data for regression
            logger.warning(f"Insufficient data for pair {pair}")
            return {}
        
        # Verify 'Close' column exists in both dataframes
        for df, t in [(df1, ticker1), (df2, ticker2)]:
            if 'Close' not in df.columns:
                logger.error(f"'Close' column not found in {t}")
                # Return empty signals as we cannot proceed
                return {ticker1: pd.Series(0, index=common_idx), 
                        ticker2: pd.Series(0, index=common_idx)}
        
        try:
            # Use fixed dimensions for all parameters
            price1 = df1['Close'].values
            price2 = df2['Close'].values
            kf = KalmanFilter(
                transition_matrices=np.eye(1),
                observation_matrices=np.ones((1, 1)),  # Fixed observation matrix
                initial_state_mean=0,
                initial_state_covariance=1.0,
                observation_covariance=1.0,
                transition_covariance=0.01
            )
            
            # Run filter with explicit dimensions
            state_means, state_covs = kf.filter(price1.reshape(-1, 1))
            hedge_ratio = state_means.flatten()
                
            # Calculate spread
            spread = price1.flatten() - hedge_ratio * price2.flatten()
            
            # Z-score of spread
            spread_series = pd.Series(spread, index=common_idx)
            spread_mean = spread_series.rolling(window=20).mean()
            spread_std = spread_series.rolling(window=20).std()
            z_score = (spread_series - spread_mean) / spread_std
            
            # Generate signals
            signals = {}
            signals[ticker1] = pd.Series(0, index=common_idx)
            signals[ticker2] = pd.Series(0, index=common_idx)
            
            # Entry rules
            signals[ticker1][z_score < -2.0] = 1  # Long ticker1
            signals[ticker1][z_score > 2.0] = -1  # Short ticker1
            
            # Opposite for ticker2
            signals[ticker2][z_score < -2.0] = -1  # Short ticker2
            signals[ticker2][z_score > 2.0] = 1  # Long ticker2
            
            logger.info(f"Generated statistical arbitrage signals for pair {pair}")
        
        except Exception as e:
            logger.error(f"Error in statistical arbitrage strategy: {e}")
            logger.info("Returning empty signals for statistical arbitrage")
            signals = {ticker1: pd.Series(0, index=common_idx), 
                      ticker2: pd.Series(0, index=common_idx)}
            
        return signals


class CombinedStrategy(Strategy):
    """Combined strategy using multiple sub-strategies."""
    def __init__(self, config, data_handler):
        super().__init__(config, data_handler)
        self.mean_reversion = MeanReversionStrategy(config, data_handler)
        self.momentum = MomentumStrategy(config, data_handler)
        self.stat_arb = StatisticalArbitrageStrategy(config, data_handler)
        self.model_trainer = ModelTrainer(config)
        self.weights = {
            'mean_reversion': 0.3,
            'momentum': 0.3,
            'ml': 0.3,
            'stat_arb': 0.1
        }
    
    def generate_signals(self, features):
        # Get signals from sub-strategies
        mean_reversion_signals = self.mean_reversion.generate_signals(features)
        momentum_signals = self.momentum.generate_signals(features)
        
        # Try to get statistical arbitrage signals, but handle errors
        try:
            stat_arb_signals = self.stat_arb.generate_signals(features)
        except Exception as e:
            logger.error(f"Error generating statistical arbitrage signals: {e}")
            # Create empty signals for all tickers
            stat_arb_signals = {ticker: pd.Series(0, index=features[ticker].index) 
                              for ticker in self.data_handler.tickers 
                              if ticker in features}
        
        # Get ML predictions
        ml_signals = {}
        for ticker in self.data_handler.tickers:
            if ticker in features and ticker in self.model_trainer.models:
                try:
                    ml_signals[ticker] = self.model_trainer.predict(ticker, features[ticker])
                except Exception as e:
                    logger.error(f"Error generating ML signals for {ticker}: {e}")
                    ml_signals[ticker] = pd.Series(0, index=features[ticker].index)
        
        # Combine signals with weights
        combined_signals = {}
        for ticker in self.data_handler.tickers:
            if ticker not in features:
                continue
                
            # Get all available signals for this ticker
            signals = {}
            if ticker in mean_reversion_signals:
                signals['mean_reversion'] = mean_reversion_signals[ticker]
            if ticker in momentum_signals:
                signals['momentum'] = momentum_signals[ticker]
            if ticker in ml_signals:
                signals['ml'] = ml_signals[ticker]
            if ticker in stat_arb_signals:
                signals['stat_arb'] = stat_arb_signals[ticker]
            
            # If no signals available, skip
            if not signals:
                logger.warning(f"No signals available for {ticker}")
                continue
                
            # Create a DataFrame for weighted combination
            signals_df = pd.DataFrame(signals)
            
            # Calculate weighted sum
            weighted_signals = pd.Series(0, index=signals_df.index)
            for strategy, weight in self.weights.items():
                if strategy in signals_df.columns:
                    weighted_signals += weight * signals_df[strategy].fillna(0)
            
            # Threshold for final signals
            combined_signals[ticker] = pd.Series(0, index=weighted_signals.index)
            combined_signals[ticker][weighted_signals > 0.5] = 1
            combined_signals[ticker][weighted_signals < -0.5] = -1
        
        logger.info(f"Generated combined signals for {len(combined_signals)} tickers")
        return combined_signals
    
    def update_weights(self, performance_metrics):
        """Update strategy weights based on recent performance."""
        if not performance_metrics:
            return
        
        # Example: If we have Sharpe ratios for each strategy
        if all(f'{strategy}_sharpe' in performance_metrics for strategy in self.weights):
            # Get sharpe ratios
            sharpes = {strategy: performance_metrics[f'{strategy}_sharpe'] for strategy in self.weights}
            
            # Normalize sharpe ratios (ensure they're positive)
            min_sharpe = min(sharpes.values())
            adjusted_sharpes = {s: max(0.1, v - min_sharpe + 0.1) for s, v in sharpes.items()}
            
            # Calculate weights proportional to sharpe ratios
            total = sum(adjusted_sharpes.values())
            self.weights = {s: v / total for s, v in adjusted_sharpes.items()}
            
            logger.info(f"Updated strategy weights: {self.weights}")


class RiskManager:
    """Handles position sizing and risk management."""
    def __init__(self, config):
        self.config = config
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.last_reset_day = datetime.now().day
        self.last_reset_week = datetime.now().isocalendar()[1]  # Week number
        self.positions = {}
    
    def calculate_position_size(self, ticker, signal, price, volatility, account_value):
        """Calculate position size based on risk per trade."""
        if signal == 0:
            return 0
        
        # Risk amount in dollars
        risk_amount = account_value * self.config.risk_per_trade
        
        # Use volatility to determine stop loss distance
        stop_distance = volatility * 2  # 2 standard deviations
        
        # Calculate shares to buy/sell
        if stop_distance > 0:
            shares = int(risk_amount / (price * stop_distance))
        else:
            shares = 0
        
        # Limit position by sector exposure
        max_position_value = account_value * self.config.max_sector_exposure
        if price * shares > max_position_value:
            shares = int(max_position_value / price)
        
        return shares if signal > 0 else -shares
    
    def update_pnl(self, daily_pnl):
        """Update daily and weekly P&L tracking."""
        # Check if we need to reset tracking
        current_day = datetime.now().day
        current_week = datetime.now().isocalendar()[1]
        
        if current_day != self.last_reset_day:
            self.daily_pnl = 0
            self.last_reset_day = current_day
        
        if current_week != self.last_reset_week:
            self.weekly_pnl = 0
            self.last_reset_week = current_week
        
        # Update P&L
        self.daily_pnl += daily_pnl
        self.weekly_pnl += daily_pnl
    
    def check_risk_limits(self, account_value):
        """Check if we've hit risk limits."""
        # Calculate percentage loss
        daily_loss_pct = self.daily_pnl / account_value if account_value > 0 else 0
        weekly_loss_pct = self.weekly_pnl / account_value if account_value > 0 else 0
        
        # Check limits
        if daily_loss_pct <= -self.config.max_daily_loss:
            logger.warning(f"Daily loss limit reached: {daily_loss_pct:.2%}. Halting trading for today.")
            return {"trading_allowed": False, "reason": "daily_loss"}
        
        if weekly_loss_pct <= -self.config.max_weekly_loss:
            logger.warning(f"Weekly loss limit reached: {weekly_loss_pct:.2%}. Halting trading for the week.")
            return {"trading_allowed": False, "reason": "weekly_loss"}
        
        return {"trading_allowed": True}


class ExecutionEngine:
    """Handles order execution via Alpaca API."""
    def __init__(self, config, api):
        self.config = config
        self.api = api
    
    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(3))
    def place_order(self, ticker, quantity, side):
        """Place order via Alpaca API with retry logic."""
        try:
            if quantity == 0:
                logger.info(f"No order placed for {ticker} - quantity is 0")
                return None
            
            order = self.api.submit_order(
                symbol=ticker,
                qty=abs(quantity),
                side='buy' if quantity > 0 else 'sell',
                type='market',
                time_in_force='day'
            )
            logger.info(f"Order placed for {ticker}: {side} {abs(quantity)} shares")
            return order
        except Exception as e:
            logger.error(f"Error placing order for {ticker}: {e}")
            raise
    
    def get_current_positions(self):
        """Get current positions from Alpaca."""
        try:
            positions = self.api.list_positions()
            return {p.symbol: {'qty': int(p.qty), 'market_value': float(p.market_value)} for p in positions}
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def get_account_value(self):
        """Get account equity value from Alpaca."""
        try:
            account = self.api.get_account()
            return float(account.equity)
        except Exception as e:
            logger.error(f"Error getting account value: {e}")
            return 0.0


class Backtester:
    """Backtests trading strategies over historical data."""
    def __init__(self, config, data_handler, feature_engineering, model_trainer, strategy_class):
        self.config = config
        self.data_handler = data_handler
        self.feature_engineering = feature_engineering
        self.model_trainer = model_trainer
        self.strategy_class = strategy_class
        self.initial_capital = config.initial_capital
        self.results = {}
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run backtest over specified period."""
        # Setup dates
        if start_date is None:
            start_date = pd.Timestamp('2022-01-01')
        if end_date is None:
            end_date = pd.Timestamp.now()
        
        logger.info(f"Running backtest from {start_date} to {end_date}")
        
        # Get data and features
        data = self.data_handler.fetch_historical_data()
        fundamental_data = self.data_handler.fetch_fundamental_data()
        
        # Calculate features
        features = self.feature_engineering.calculate_technical_indicators()
        features = self.feature_engineering.add_fundamental_features(fundamental_data)
        selected_features = self.feature_engineering.select_features()
        
        # Train models on data before the backtest period
        for ticker, df in selected_features.items():
            training_data = df[df.index < start_date].copy()
            if len(training_data) > 100:  # Ensure enough data for training
                self.model_trainer.train_model(ticker, training_data)
        
        # Initialize strategy
        strategy = self.strategy_class(self.config, self.data_handler)
        
        # Initialize portfolio tracking
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'equity': [self.initial_capital],
            'dates': [start_date],
            'trades': []
        }
        
        # Run backtest day by day
        date_range = pd.date_range(start=start_date, end=end_date, freq='B')  # Business days
        
        for date in date_range:
            try:
                # Get data for this date
                current_features = {}
                for ticker, df in selected_features.items():
                    if date in df.index:
                        # Get current and past data (no lookahead bias)
                        current_features[ticker] = df[df.index <= date].copy()
                
                # Skip if no data
                if not current_features:
                    continue
                
                # Generate signals
                signals = strategy.generate_signals(current_features)
                
                # Execute trades
                self._execute_backtest_trades(signals, current_features, portfolio, date)
                
                # Calculate portfolio value
                portfolio_value = portfolio['cash']
                for ticker, position in portfolio['positions'].items():
                    if position['shares'] != 0:
                        # Safety check for accessing price data
                        if (ticker in current_features and 
                            date in current_features[ticker].index and 
                            'Close' in current_features[ticker].columns):
                            current_price = current_features[ticker].loc[date, 'Close']
                        else:
                            current_price = position['price']  # Use last known price
                        
                        portfolio_value += position['shares'] * current_price
                
                # Record equity curve
                portfolio['equity'].append(portfolio_value)
                portfolio['dates'].append(date)
                
            except Exception as e:
                logger.error(f"Error in backtest loop for date {date}: {e}")
                # Continue the backtest despite the error
                continue
        
        # Calculate performance metrics
        self.results = self._calculate_performance_metrics(portfolio)
        
        return self.results
    
    def _execute_backtest_trades(self, signals, features, portfolio, date):
        """Execute trades in backtest."""
        for ticker, signal_series in signals.items():
            if date not in signal_series.index:
                continue
            
            signal = signal_series.loc[date]
            
            # Skip if no signal
            if signal == 0:
                continue
            
            # Get current price
            if ticker in features and date in features[ticker].index:
                # Verify 'Close' column exists
                if 'Close' not in features[ticker].columns:
                    logger.error(f"'Close' column not found for {ticker} in backtest")
                    continue
                    
                current_price = features[ticker].loc[date, 'Close']
                
                # Check for 'Volatility_20d' column
                if 'Volatility_20d' in features[ticker].columns:
                    current_volatility = features[ticker].loc[date, 'Volatility_20d']
                else:
                    current_volatility = 0.02  # Default volatility
            else:
                continue
            
            # Calculate position size (risk management)
            portfolio_value = portfolio['cash']
            for t, pos in portfolio['positions'].items():
                if t in features and date in features[t].index and 'Close' in features[t].columns:
                    current_price_pos = features[t].loc[date, 'Close']
                    portfolio_value += pos['shares'] * current_price_pos
            
            risk_per_trade = self.config.risk_per_trade
            risk_amount = portfolio_value * risk_per_trade
            
            # Use volatility for stop loss distance
            stop_distance = max(0.01, current_volatility * 2)  # At least 1%
            
            # Calculate shares
            shares = int(risk_amount / (current_price * stop_distance))
            
            # Limit by available cash
            max_shares = int(portfolio['cash'] / current_price) if signal > 0 else float('inf')
            shares = min(shares, max_shares)
            
            # Apply signal direction
            shares = shares if signal > 0 else -shares
            
            # Execute trade
            cost = shares * current_price
            
            # Update positions
            if ticker not in portfolio['positions']:
                portfolio['positions'][ticker] = {'shares': 0, 'price': 0}
            
            # Record trade
            trade = {
                'date': date,
                'ticker': ticker,
                'action': 'buy' if shares > 0 else 'sell',
                'shares': abs(shares),
                'price': current_price,
                'cost': cost
            }
            portfolio['trades'].append(trade)
            
            # Update portfolio
            current_shares = portfolio['positions'][ticker]['shares']
            new_shares = current_shares + shares
            
            # If closing or reversing position
            if current_shares * new_shares <= 0:  # Sign change or one is zero
                # Calculate P&L if closing
                if current_shares != 0:
                    entry_price = portfolio['positions'][ticker]['price']
                    pnl = current_shares * (current_price - entry_price)
                    logger.info(f"Backtest: Closed position in {ticker} with P&L: ${pnl:.2f}")
                
                # Reset position if closing completely
                if new_shares == 0:
                    portfolio['positions'][ticker] = {'shares': 0, 'price': 0}
                else:
                    # New position
                    portfolio['positions'][ticker] = {
                        'shares': new_shares,
                        'price': current_price
                    }
            else:
                # Adding to position, update average price
                total_cost = (current_shares * portfolio['positions'][ticker]['price']) + cost
                total_shares = current_shares + shares
                avg_price = total_cost / total_shares if total_shares != 0 else 0
                
                portfolio['positions'][ticker] = {
                    'shares': total_shares,
                    'price': avg_price
                }
            
            # Update cash
            portfolio['cash'] -= cost
    
    def _calculate_performance_metrics(self, portfolio):
        """Calculate performance metrics from backtest results."""
        try:
            # Safety check for valid portfolio data
            if not portfolio['equity'] or len(portfolio['equity']) < 2:
                logger.error("Insufficient equity data to calculate performance metrics")
                return {
                    'initial_capital': self.initial_capital,
                    'final_capital': self.initial_capital,
                    'total_return': 0.0,
                    'annual_return': 0.0,
                    'volatility': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'win_rate': 0.0,
                    'trades': 0,
                    'equity_curve': pd.Series([self.initial_capital], index=[pd.Timestamp.now()])
                }
            
            # Convert to pandas Series for calculations
            equity_curve = pd.Series(portfolio['equity'], index=portfolio['dates'])
            
            # Calculate returns
            returns = equity_curve.pct_change().dropna()
            
            # Calculate metrics with safety checks
            total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1 if len(equity_curve) > 1 else 0.0
            
            # Annual return needs enough data
            if len(returns) > 10:
                annual_return = (1 + total_return) ** (252 / len(returns)) - 1
            else:
                annual_return = 0.0
                
            # Volatility needs enough data points
            if len(returns) > 5:
                volatility = returns.std() * np.sqrt(252)
            else:
                volatility = 0.0
                
            # Sharpe ratio needs valid volatility
            if volatility > 0:
                sharpe_ratio = (annual_return - 0.01) / volatility
            else:
                sharpe_ratio = 0.0
            
            # Calculate drawdown
            peak = equity_curve.cummax()
            drawdown = (equity_curve - peak) / peak
            max_drawdown = drawdown.min()
            
            # Win rate calculation
            win_rate = 0.0
            if portfolio['trades']:
                try:
                    trades_df = pd.DataFrame(portfolio['trades'])
                    if len(trades_df) > 0:
                        # Calculate P&L for trades
                        trades_df['pnl'] = 0
                        tickers = trades_df['ticker'].unique()
                        
                        for ticker in tickers:
                            ticker_trades = trades_df[trades_df['ticker'] == ticker].copy()
                            
                            # Track position and cost basis
                            position = 0
                            cost_basis = 0
                            
                            # Process trades in order
                            for idx, row in ticker_trades.iterrows():
                                if row['action'] == 'buy':
                                    # Update position and cost basis
                                    new_position = position + row['shares']
                                    if position > 0:
                                        # Weighted average cost basis
                                        cost_basis = ((position * cost_basis) + (row['shares'] * row['price'])) / new_position
                                    else:
                                        cost_basis = row['price']
                                    position = new_position
                                else:  # sell
                                    # Calculate P&L for the sell
                                    if position > 0:
                                        pnl = (row['price'] - cost_basis) * min(position, row['shares'])
                                        trades_df.loc[idx, 'pnl'] = pnl
                                    position = max(0, position - row['shares'])
                        
                        # Calculate win rate
                        winning_trades = len(trades_df[trades_df['pnl'] > 0])
                        total_trades = len(trades_df[trades_df['pnl'] != 0])
                        win_rate = winning_trades / total_trades if total_trades > 0 else 0
                except Exception as e:
                    logger.error(f"Error calculating win rate: {e}")
                    win_rate = 0.0
            
            # Return metrics
            metrics = {
                'initial_capital': portfolio['equity'][0],
                'final_capital': portfolio['equity'][-1],
                'total_return': total_return,
                'annual_return': annual_return,
                'volatility': volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'trades': len(portfolio['trades']),
                'equity_curve': equity_curve
            }
            
            logger.info(f"Backtest results: {metrics['total_return']:.2%} return, {metrics['sharpe_ratio']:.2f} Sharpe, {metrics['max_drawdown']:.2%} max drawdown")
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {e}")
            # Return default metrics to prevent crashes
            return {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_return': 0.0,
                'annual_return': 0.0,
                'volatility': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'trades': 0,
                'equity_curve': pd.Series([self.initial_capital], index=[pd.Timestamp.now()])
            }
    
def plot_results(self):
    """Plot backtest results."""
    if not self.results or 'equity_curve' not in self.results:
        logger.error("No backtest results to plot")
        return
    
    plt.figure(figsize=(12, 8))
    
    # Convert timestamps to numerical indices to avoid errors
    x_indices = range(len(self.results['equity_curve']))
    
    # Equity curve
    plt.subplot(2, 1, 1)
    plt.plot(x_indices, self.results['equity_curve'].values)
    plt.title('Portfolio Equity Curve')
    plt.grid(True)
    
    # Drawdown
    plt.subplot(2, 1, 2)
    peak = self.results['equity_curve'].cummax()
    drawdown = (self.results['equity_curve'] - peak) / peak
    plt.fill_between(x_indices, drawdown.values, 0, color='red', alpha=0.3)
    plt.title('Drawdown')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('backtest_results.png')
    plt.close()

class TradingSystem:
    """Main trading system that integrates all components."""
    def __init__(self, config_file='config.json'):
        self.config = Config(config_file)
        self.data_handler = DataHandler(self.config)
        self.api = self.data_handler.setup_alpaca_api()
        self.feature_engineering = FeatureEngineering(self.data_handler)
        self.model_trainer = ModelTrainer(self.config)
        self.risk_manager = RiskManager(self.config)
        self.execution_engine = ExecutionEngine(self.config, self.api)
        
        # Select strategy based on configuration
        strategy_map = {
            'mean_reversion': MeanReversionStrategy,
            'momentum': MomentumStrategy,
            'stat_arb': StatisticalArbitrageStrategy,
            'combined': CombinedStrategy
        }
        self.strategy_class = strategy_map.get(self.config.selected_strategy, CombinedStrategy)
        self.strategy = self.strategy_class(self.config, self.data_handler)
        
        self.backtester = Backtester(
            self.config, 
            self.data_handler,
            self.feature_engineering,
            self.model_trainer,
            self.strategy_class
        )
    
    def initialize(self):
        """Initialize the trading system."""
        logger.info("Initializing trading system")
        
        # Fetch historical data
        self.data_handler.fetch_historical_data()
        
        # CRITICAL: Print data structure for debugging
        for ticker, df in self.data_handler.data.items():
            if df is not None and not df.empty:
                print(f"\nData for {ticker}:")
                print(f"Data type: {type(df)}")
                print(f"Column type: {type(df.columns)}")
                print(f"Columns: {df.columns.tolist()}")
                print(f"First row: {df.iloc[0].to_dict()}")
        
        # Check if we have data for all tickers
        missing_data = [ticker for ticker, data in self.data_handler.data.items() 
                         if data is None or data.empty]
        if missing_data:
            logger.error(f"Missing data for the following tickers: {missing_data}")
            if len(missing_data) == len(self.data_handler.tickers):
                logger.error("No data available for any ticker. Cannot initialize system.")
                return False
        
        # Fetch fundamental data
        fundamental_data = self.data_handler.fetch_fundamental_data()
        
        # Calculate features
        features = self.feature_engineering.calculate_technical_indicators()
        features = self.feature_engineering.add_fundamental_features(fundamental_data)
        
        # Check if features were calculated correctly
        if not features:
            logger.error("Failed to calculate features")
            return False
            
        # Select features
        selected_features = self.feature_engineering.select_features()
        
        # Check if feature selection was successful
        if not selected_features:
            logger.error("Feature selection failed")
            return False
        
        # Train initial models
        success_count = 0
        for ticker, df in selected_features.items():
            if df is not None and not df.empty:
                model, accuracy = self.model_trainer.train_model(ticker, df)
                if model is not None:
                    success_count += 1
                    self.model_trainer.train_alternative_models(ticker, df)
        
        if success_count == 0:
            logger.error("Failed to train any models")
            return False
            
        logger.info(f"Trading system initialized successfully with {success_count} models")
        return True
    
    def run_backtest(self, start_date=None, end_date=None):
        """Run backtest over specified period."""
        logger.info("Running backtest")
        results = self.backtester.run_backtest(start_date, end_date)
        #self.backtester.plot_results()
        return results
    
    def run_trading_iteration(self):
        """Run a single trading iteration."""
        logger.info("Running trading iteration")
        
        # Check if market is open
        try:
            clock = self.api.get_clock()
            if not clock.is_open:
                logger.info("Market is closed. Skipping trading iteration.")
                return False
        except Exception as e:
            logger.error(f"Error checking market status: {e}")
            return False
        
        # Check risk limits
        account_value = self.execution_engine.get_account_value()
        risk_check = self.risk_manager.check_risk_limits(account_value)
        if not risk_check['trading_allowed']:
            logger.warning(f"Trading halted due to {risk_check['reason']}")
            return False
        
        # Fetch latest data
        live_data = self.data_handler.fetch_live_data(self.api)
        
        # Update features with live data
        features = {}
        for ticker, df in live_data.items():
            # Merge with historical data if needed for indicators
            if ticker in self.data_handler.data:
                hist_data = self.data_handler.data[ticker].copy()
                combined_data = pd.concat([hist_data, df]).drop_duplicates()
                
                # Update features
                temp_features = FeatureEngineering({'data': {ticker: combined_data}})
                calc_features = temp_features.calculate_technical_indicators()
                
                # Get only the latest data
                features[ticker] = calc_features[ticker].iloc[-100:]  # Last 100 rows for calculation
            
        # Check if retraining is needed
        performance_metrics = {
            'sharpe_ratio': self.backtester.results.get('sharpe_ratio', 2.0),  # Placeholder if no backtest run yet
            'consecutive_loss_days': 0  # Placeholder, would track this in practice
        }
        
        if self.model_trainer.should_retrain(performance_metrics):
            logger.info("Retraining models")
            # Fetch fresh data for training
            self.data_handler.fetch_historical_data()
            fundamental_data = self.data_handler.fetch_fundamental_data()
            
            # Calculate features
            features_for_training = self.feature_engineering.calculate_technical_indicators()
            features_for_training = self.feature_engineering.add_fundamental_features(fundamental_data)
            selected_features = self.feature_engineering.select_features()
            
            # Retrain models
            for ticker, df in selected_features.items():
                self.model_trainer.train_model(ticker, df)
            
            # Update last training date
            self.model_trainer.last_training_date = datetime.now()
        
        # Generate trading signals
        signals = self.strategy.generate_signals(features)
        
        # Execute trades
        current_positions = self.execution_engine.get_current_positions()
        
        for ticker, signal_series in signals.items():
            if len(signal_series) == 0:
                continue
            
            # Get latest signal
            latest_signal = signal_series.iloc[-1]
            
            # Skip if no signal
            if latest_signal == 0:
                continue
            
            # Get current price and volatility
            current_price = features[ticker]['Close'].iloc[-1]
            current_volatility = features[ticker]['Volatility_20d'].iloc[-1] if 'Volatility_20d' in features[ticker].columns else 0.02
            
            # Calculate position size
            position_size = self.risk_manager.calculate_position_size(
                ticker, 
                latest_signal, 
                current_price, 
                current_volatility, 
                account_value
            )
            
            # Adjust for existing position
            current_position = current_positions.get(ticker, {'qty': 0})['qty']
            target_position = position_size
            order_size = target_position - current_position
            
            # Place order if needed
            if order_size != 0:
                side = 'buy' if order_size > 0 else 'sell'
                self.execution_engine.place_order(ticker, order_size, side)
        
        logger.info("Trading iteration completed")
        return True
    
    def run_paper_trading(self, days=30):
        """Run paper trading for a specified number of days."""
        logger.info(f"Starting paper trading for {days} days")
        
        # Initialize system
        self.initialize()
        
        # Run backtest to verify strategy
        start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        backtest_results = self.run_backtest(start_date, end_date)
        
        # Check if strategy is viable
        if backtest_results['sharpe_ratio'] < 0.5:
            logger.warning(f"Backtest results below threshold: Sharpe ratio {backtest_results['sharpe_ratio']:.2f}")
            return False
        
        # Schedule trading iterations
        start_time = datetime.now()
        end_time = start_time + timedelta(days=days)
        
        while datetime.now() < end_time:
            # Run trading iteration
            success = self.run_trading_iteration()
            
            # Wait for next iteration (e.g., every 15 minutes during market hours)
            time.sleep(900)  # 15 minutes
            
            # Check if we need to save state
            if datetime.now().hour == 16 and datetime.now().minute < 15:  # Around market close
                self.save_state()
        
        logger.info("Paper trading completed")
        return True
    
    def save_state(self, state_file='trading_state.json'):
        """Save current state of the trading system."""
        state = {
            'last_training_date': self.model_trainer.last_training_date.strftime('%Y-%m-%d'),
            'daily_pnl': self.risk_manager.daily_pnl,
            'weekly_pnl': self.risk_manager.weekly_pnl,
            'last_reset_day': self.risk_manager.last_reset_day,
            'last_reset_week': self.risk_manager.last_reset_week,
            'strategy_weights': getattr(self.strategy, 'weights', {})
        }
        
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=4)
        
        logger.info(f"Trading state saved to {state_file}")
    
    def load_state(self, state_file='trading_state.json'):
        """Load saved state of the trading system."""
        if os.path.exists(state_file):
            with open(state_file, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.model_trainer.last_training_date = datetime.strptime(state['last_training_date'], '%Y-%m-%d')
            self.risk_manager.daily_pnl = state['daily_pnl']
            self.risk_manager.weekly_pnl = state['weekly_pnl']
            self.risk_manager.last_reset_day = state['last_reset_day']
            self.risk_manager.last_reset_week = state['last_reset_week']
            
            # Restore strategy weights if applicable
            if hasattr(self.strategy, 'weights') and 'strategy_weights' in state:
                self.strategy.weights = state['strategy_weights']
            
            logger.info(f"Trading state loaded from {state_file}")
            return True
        
        logger.warning(f"No state file found at {state_file}")
        return False


def main():
    """Main function to run the trading system."""
    print("-------------------------------------------------------")
    print("ML Trading System for AAPL and BA")
    print("-------------------------------------------------------")
    print("Before running, please ensure you've set up your config.json file")
    print("with your Alpaca API keys if you want to use live trading.")
    print("-------------------------------------------------------")
    
    # Create trading system
    trading_system = TradingSystem()
    
    # Try a direct test fetch first to diagnose the issue
    print("Testing direct yfinance fetch to diagnose data issues...")
    try:
        # Try single ticker fetch first
        for ticker in ["AAPL", "BA"]:
            test_data = yf.download(ticker, start='2023-01-01', end='2023-12-31', progress=False)
            print(f"\nDirect yfinance test for {ticker}:")
            print(f"Columns type: {type(test_data.columns)}")
            print(f"Columns: {test_data.columns.tolist()}")
            print(f"Shape: {test_data.shape}")
            print(f"First few rows: {test_data.head(2)}")
        
        # Try batch fetch
        test_batch = yf.download(["AAPL", "BA"], start='2023-01-01', end='2023-12-31', progress=False, group_by='ticker')
        print("\nDirect yfinance batch test:")
        print(f"Columns type: {type(test_batch.columns)}")
        print(f"Columns structure: {test_batch.columns}")
        if isinstance(test_batch.columns, pd.MultiIndex):
            print(f"Level 0: {test_batch.columns.get_level_values(0).unique().tolist()}")
            print(f"Level 1: {test_batch.columns.get_level_values(1).unique().tolist()}")
            # Extract AAPL data
            aapl_data = test_batch['AAPL']
            print(f"AAPL extracted columns: {aapl_data.columns.tolist()}")
    except Exception as e:
        print(f"Test fetch error: {e}")
    
    # Initialize system
    print("\nInitializing trading system...")
    init_success = trading_system.initialize()
    if not init_success:
        print("Initialization had serious issues - cannot proceed with backtest.")
        return
    
    try:
        # Check final data structure before proceeding
        print("\nFinal data verification:")
        for ticker, df in trading_system.data_handler.data.items():
            if df is not None and not df.empty:
                print(f"{ticker} data shape: {df.shape}, columns: {df.columns.tolist()}")
                has_required = all(col in df.columns for col in ['Open', 'High', 'Low', 'Close', 'Volume'])
                print(f"{ticker} has all required columns: {has_required}")
        
        # Check feature data 
        print("\nFeature verification:")
        for ticker, df in trading_system.feature_engineering.features.items():
            if df is not None and not df.empty:
                features_count = len(df.columns)
                has_target = 'Target' in df.columns
                print(f"{ticker} features count: {features_count}, has Target column: {has_target}")
        
        # Set backtest period
        start_date = '2020-01-01'
        end_date = datetime.now().strftime('%Y-%m-%d')
        print(f"\nRunning backtest from {start_date} to {end_date}...")
        
        # Run backtest
        backtest_results = trading_system.run_backtest(start_date, end_date)
        
        # Display backtest results
        print("\nBacktest Results:")
        print(f"Total Return: {backtest_results['total_return']:.2%}")
        print(f"Annual Return: {backtest_results['annual_return']:.2%}")
        print(f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        print(f"Win Rate: {backtest_results['win_rate']:.2%}")
        print(f"Number of Trades: {backtest_results['trades']}")
        
        # Check if Alpaca API is configured for paper trading
        if trading_system.api is None:
            print("\nPaper trading requires Alpaca API. Please configure your API keys in config.json")
            print("and restart the program if you want to run paper trading.")
        else:
            # Run paper trading for 30 days
            trade_decision = input("\nDo you want to start paper trading? (y/n): ")
            if trade_decision.lower() == 'y':
                trading_system.run_paper_trading(days=30)
    
    except Exception as e:
        print(f"\nError during operation: {e}")
        print("Please check logs for more details and verify your data sources.")
        logger.error(f"Operation error: {e}", exc_info=True)
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()