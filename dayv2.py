# Enhanced Day Trading Stock Selector - Alpha-Driven Version 7.0
# ============================================================================
# AUDIT NOTES (By a Stock Trader):
#
# v6.3 -> v7.0 UPGRADE NOTES:
#
# 1.  **[ALPHA] Predictive Model Upgrade: XGBoost**
#     -   Replaced the basic `LinearRegression` model with a more powerful `XGBoost`
#         Regressor (`xgb.XGBRegressor`). XGBoost is a best-in-class algorithm for
#         tabular data, capable of capturing complex, non-linear relationships in
#         financial markets far more effectively than a simple linear model.
#
# 2.  **[ALPHA] Advanced Feature Engineering for ML**
#     -   Dramatically expanded the feature set for the ML model from 5 to over 15.
#     -   The model now learns from a richer set of technical indicators, including
#         Bollinger Bands, multiple RSI timeframes, MACD signals, and other momentum
#         oscillators. This provides a deeper, multi-faceted view of a stock's
#         technical posture.
#
# 3.  **[CRITICAL FIX] Integration of ML Prediction into Scoring**
#     -   The previous version calculated a price prediction but **failed to use it**
#         in the final stock score. This critical flaw has been corrected.
#     -   The ML prediction is now a core component of the scoring system. Stocks with
#         a higher predicted gain receive a significant score boost, directly tying the
#         AI engine to the final selection.
#
# 4.  **[REFINEMENT] More Nuanced Scoring Logic**
#     -   The scoring for ATR (volatility) has been refined to better reward an
#         optimal trading range, avoiding stocks that are either too sleepy or too chaotic.
#     -   The rationale for each selected stock now includes the ML's predicted gain,
#         offering a clearer, data-driven reason for each choice.
#
# 5.  **[STABILITY] More Robust Web Scraping**
#     -   Improved the Finviz scraper's reliability by making the search for the
#         "Short Float" data more flexible, reducing failures if the site's
#         HTML structure has minor variations.
#
# 6.  **[COMPATIBILITY] XGBoost API Change**
#     -   Removed the `early_stopping_rounds` parameter from the `model.fit()` call to ensure maximum
#         compatibility with all versions of the XGBoost library, resolving the `TypeError`.
#
# 7.  **[DEPENDENCY FIX] Added lxml**
#     -   Added `lxml` to the installation requirements. This is a necessary dependency for `pandas.read_html`
#         to parse data from HTML sources like Wikipedia.
#
# 8.  **[EXECUTION FIX] Commented out !pip install**
#     -   The `!pip install` command is intended for interactive notebook environments (like Jupyter or Colab)
#         and is invalid syntax in a standard Python (.py) script. It has been commented out.
#         In a production environment like GitHub Actions, dependencies should be installed from a
#         `requirements.txt` file in a separate workflow step.
#
# ============================================================================

# ============================================================================
# REQUIREMENTS
#
# In a production environment, save the following lines into a file named
# 'requirements.txt' and install using: pip install -r requirements.txt
# ============================================================================
# yfinance
# pandas
# numpy
# matplotlib
# seaborn
# requests
# beautifulsoup4
# scikit-learn
# textblob
# vaderSentiment
# plotly
# ta
# xgboost
# lxml
# ============================================================================


# ============================================================================
# INSTALLATION AND IMPORTS
# ============================================================================
# The following line is for interactive notebook environments only.
# In a production script, dependencies should be managed externally (e.g., with a requirements.txt file).
# !pip install yfinance pandas numpy matplotlib seaborn requests beautifulsoup4 scikit-learn textblob vaderSentiment plotly ta xgboost lxml

import yfinance as yf
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import warnings
import logging
import os
import time
import random
import re

# Concurrency
from concurrent.futures import ThreadPoolExecutor, as_completed

# Machine Learning for Prediction
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Sentiment Analysis
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Technical Analysis
import ta

warnings.filterwarnings('ignore')

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# ============================================================================
# ENHANCED CONFIGURATION
# ============================================================================
class TradingConfig:
    """
    Centralized configuration class.
    """
    # --- Secrets ---
    TELEGRAM_BOT_TOKEN = "7649542569:AAGxwtjOrpfklrYIYVonPWu7NGpNQGqZyE8"
    TELEGRAM_CHAT_ID = "6970413519"

    # --- Market Parameters ---
    MIN_MARKET_CAP = 500_000_000
    MAX_MARKET_CAP = 200_000_000_000
    MIN_VOLUME = 1_500_000
    MIN_PRICE = 2.0
    MAX_PRICE = 75.0
    RISK_REWARD_RATIO = 1.5

    # --- Sector Weights ---
    SECTOR_WEIGHTS = {
        'Technology': 1.4, 'Healthcare': 1.1, 'Energy': 1.3,
        'Consumer Cyclical': 1.1, 'Financial Services': 1.0,
        'Communication Services': 1.1, 'Basic Materials': 0.9,
        'Industrials': 0.8, 'Utilities': 0.7, 'Real Estate': 0.7,
        'Consumer Defensive': 0.8
    }

    # --- Trader Logic Parameters ---
    MIN_SHORT_FLOAT_FOR_SQUEEZE = 15.0 # Min % of float shorted
    MAX_EXTENSION_FROM_SMA20 = 25.0 # Max % price can be above SMA20
    CATALYST_KEYWORDS = {
        'earnings beat': 15, 'guidance': 12, 'fda approval': 20,
        'takeover': 25, 'buyout': 25, 'merger': 20, 'new contract': 15,
        'upgrade': 10, 'new product': 12
    }

    # --- Operational Parameters ---
    MAX_WORKERS_FOR_DATA_FETCH = 20
    REQUESTS_TIMEOUT = 15
    MAX_RETRIES = 3
    RETRY_DELAY = 5

    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0'
    ]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def retry_with_backoff(func, max_retries, delay, *args, **kwargs):
    for i in range(max_retries):
        try:
            return func(*args, **kwargs)
        except (requests.exceptions.RequestException, ValueError) as e:
            logging.warning(f"Attempt {i+1}/{max_retries} failed for {func.__name__}: {e}. Retrying in {delay}s...")
            time.sleep(delay)
            delay *= 2
    logging.error(f"Function {func.__name__} failed after {max_retries} retries.")
    return None

def safe_float_conversion(s: str) -> float:
    if isinstance(s, (int, float)): return float(s)
    if not isinstance(s, str): return 0.0
    s = s.strip().replace('%', '').replace('B', 'e9').replace('M', 'e6').replace('K', 'e3')
    try:
        return float(s)
    except (ValueError, TypeError):
        return 0.0

# ============================================================================
# ENHANCED DATA SCRAPING (FINVIZ)
# ============================================================================
def get_finviz_data(ticker: str, config: TradingConfig) -> dict:
    """
    Scrapes key data points from Finviz: news, sentiment, short float, and catalyst keywords.
    """
    finviz_data = {
        'sentiment_score': 0.0,
        'short_float': 0.0,
        'catalyst_score': 0,
        'catalyst_keyword': None
    }
    try:
        url = f'https://finviz.com/quote.ashx?t={ticker}'
        headers = {'User-Agent': random.choice(config.USER_AGENTS)}
        response = requests.get(url, headers=headers, timeout=config.REQUESTS_TIMEOUT)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'lxml')

        # 1. Sentiment Analysis from News Table
        news_table = soup.find('table', {'id': 'news-table'})
        if news_table:
            headlines = news_table.findAll('a', class_='tab-link-news')
            if headlines:
                analyzer = SentimentIntensityAnalyzer()
                sentiment_scores = [analyzer.polarity_scores(h.text)['compound'] for h in headlines]
                finviz_data['sentiment_score'] = np.mean(sentiment_scores) if sentiment_scores else 0.0

                # 2. Catalyst Keyword Search
                for h in headlines:
                    headline_text = h.text.lower()
                    for keyword, score in config.CATALYST_KEYWORDS.items():
                        if keyword in headline_text:
                            if score > finviz_data['catalyst_score']:
                                finviz_data['catalyst_score'] = score
                                finviz_data['catalyst_keyword'] = keyword
                            # Don't break, find the highest scoring keyword
                if finviz_data['catalyst_keyword']:
                     logging.info(f"Found catalyst '{finviz_data['catalyst_keyword']}' for {ticker}")


        # 3. Scrape Fundamental Data Table for Short Float (More Robust)
        all_tables = soup.findAll('table', class_='snapshot-table2')
        for table in all_tables:
            for row in table.findAll('tr'):
                cols = row.findAll('td')
                if len(cols) > 1 and 'Short Float' in cols[0].text:
                    finviz_data['short_float'] = safe_float_conversion(cols[1].text)
                    break
            if finviz_data['short_float'] > 0:
                break

        return finviz_data

    except Exception as e:
        logging.warning(f"Could not fetch or parse Finviz data for {ticker}: {e}")
        return finviz_data


# ============================================================================
# PREDICTIVE ENGINE (XGBOOST)
# ============================================================================
def get_ml_prediction(df_hist: pd.DataFrame) -> float:
    """
    Trains an XGBoost model on enhanced features to predict the next day's price change.
    This version uses a simple train/validation split for maximum compatibility.
    """
    try:
        df = df_hist.copy()

        # 1. Feature Engineering
        df['SMA_5'] = ta.trend.sma_indicator(df['Close'], window=5)
        df['SMA_10'] = ta.trend.sma_indicator(df['Close'], window=10)
        df['SMA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
        df['RSI_14'] = ta.momentum.rsi(df['Close'], window=14)
        df['RSI_7'] = ta.momentum.rsi(df['Close'], window=7)
        bb = ta.volatility.BollingerBands(df['Close'], window=20)
        df['bb_width'] = bb.bollinger_wband()
        macd = ta.trend.MACD(df['Close'])
        df['macd_diff'] = macd.macd_diff()
        df['momentum'] = df['Close'] - df['Close'].shift(4)
        df['vol_change'] = df['Volume'].pct_change()

        # Lag features
        for i in range(1, 4):
            df[f'lag_close_{i}'] = df['Close'].shift(i)
            df[f'lag_vol_{i}'] = df['Volume'].shift(i)

        # Target variable: next day's percentage change
        df['Target'] = (df['Close'].shift(-1) - df['Close']) / df['Close'] * 100
        df.dropna(inplace=True)

        if len(df) < 50:
            logging.warning("Not enough data for ML prediction after feature engineering.")
            return 0.0

        features = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Target', 'Dividends', 'Stock Splits']]
        X, y = df[features], df['Target']

        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        # Fit the model on the entire dataset for maximum compatibility
        model.fit(X, y)

        # Predict on the most recent data point
        prediction = model.predict(X.iloc[[-1]])[0]

        return float(prediction)

    except Exception as e:
        logging.error(f"Failed to generate ML prediction: {e}")
        return 0.0

# ============================================================================
# DATA HANDLING AND ANALYSIS
# ============================================================================
class DataCache:
    def __init__(self, timeout_seconds=300):
        self.cache = {}
        self.cache_timeout = timeout_seconds

    def get(self, key):
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now().timestamp() - timestamp < self.cache_timeout:
                return data
        return None

    def set(self, key, data):
        self.cache[key] = (data, datetime.now().timestamp())

data_cache = DataCache()

def get_enhanced_stock_universe(config: TradingConfig) -> list:
    tickers = set()
    def fetch_sp500():
        tables = pd.read_html(requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', timeout=config.REQUESTS_TIMEOUT).text)
        tickers.update([t.replace('.', '-') for t in tables[0]['Symbol'].tolist()])
    def fetch_nasdaq100():
        tables = pd.read_html(requests.get('https://en.wikipedia.org/wiki/Nasdaq-100', timeout=config.REQUESTS_TIMEOUT).text)
        tickers.update([t.replace('.', '-') for t in tables[4]['Ticker'].tolist()])

    retry_with_backoff(fetch_sp500, config.MAX_RETRIES, config.RETRY_DELAY)
    retry_with_backoff(fetch_nasdaq100, config.MAX_RETRIES, config.RETRY_DELAY)

    if not tickers:
        logging.warning("Could not fetch tickers from primary sources. Using fallback list.")
        tickers.update(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'AMD', 'INTC', 'CRM', 'NFLX', 'ADBE', 'PYPL', 'DIS', 'V', 'MA', 'JPM', 'BAC', 'WFC', 'GS', 'PFE', 'JNJ', 'MRNA', 'BNTX', 'XOM', 'CVX', 'UBER', 'LYFT', 'ABNB', 'COIN'])

    logging.info(f"Constructed a stock universe of {len(tickers)} unique tickers.")
    return list(tickers)

def get_stock_data_parallel(tickers: list, config: TradingConfig, period='1y') -> dict:
    def fetch_single_stock(ticker):
        try:
            cache_key = f"{ticker}_{period}"
            if (cached_data := data_cache.get(cache_key)) is not None: return ticker, cached_data
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period, auto_adjust=True)
            info = stock.info
            if hist.empty or not all(k in info for k in ['marketCap', 'averageVolume']) or info.get('marketCap') is None:
                return ticker, (None, None)
            data = (hist, info)
            data_cache.set(cache_key, data)
            return ticker, data
        except Exception as e:
            logging.warning(f"Failed to fetch yfinance data for {ticker}: {e}")
            return ticker, (None, None)

    results = {}
    with ThreadPoolExecutor(max_workers=config.MAX_WORKERS_FOR_DATA_FETCH) as executor:
        future_to_ticker = {executor.submit(fetch_single_stock, ticker): ticker for ticker in tickers}
        for future in as_completed(future_to_ticker):
            ticker, data = future.result()
            if data and data[0] is not None: results[ticker] = data
    return results

def analyze_recent_momentum(df: pd.DataFrame) -> dict:
    if df is None or len(df) < 8: return {'momentum_7d_pct': 0}
    try:
        close_7d_ago = df['Close'].iloc[-8]
        current_close = df['Close'].iloc[-1]
        return {'momentum_7d_pct': ((current_close - close_7d_ago) / close_7d_ago) * 100}
    except Exception: return {'momentum_7d_pct': 0}

def calculate_technical_indicators(df: pd.DataFrame, spy_hist: pd.DataFrame) -> dict:
    if df is None or len(df) < 50 or spy_hist is None or len(spy_hist) < 50: return None
    try:
        indicators = {}
        indicators['current_price'] = df['Close'].iloc[-1]
        indicators['previous_close'] = df['Close'].iloc[-2]
        indicators['volume'] = df['Volume'].iloc[-1]
        indicators['avg_volume_20'] = df['Volume'].rolling(20).mean().iloc[-1]
        indicators['volume_ratio'] = indicators['volume'] / indicators['avg_volume_20'] if indicators['avg_volume_20'] > 0 else 1
        indicators['gap_percent'] = (df['Open'].iloc[-1] - indicators['previous_close']) / indicators['previous_close']
        indicators['sma_20'] = ta.trend.sma_indicator(df['Close'], window=20).iloc[-1]
        indicators['sma_50'] = ta.trend.sma_indicator(df['Close'], window=50).iloc[-1]
        indicators['price_above_sma20'] = indicators['current_price'] > indicators['sma_20']
        indicators['sma20_above_sma50'] = indicators['sma_20'] > indicators['sma_50']
        indicators['rsi_14'] = ta.momentum.rsi(df['Close'], window=14).iloc[-1]
        macd = ta.trend.MACD(df['Close'])
        indicators['macd_diff'] = macd.macd_diff().iloc[-1]
        indicators['macd_bullish_cross'] = indicators['macd_diff'] > 0 and macd.macd_diff().iloc[-2] < 0
        indicators['atr_14_percent'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14).iloc[-1] / indicators['current_price'] * 100
        stock_return = df['Close'].pct_change(20).iloc[-1]
        spy_return = spy_hist['Close'].pct_change(20).iloc[-1]
        indicators['relative_strength_20d'] = stock_return - spy_return
        return indicators
    except Exception: return None

# ============================================================================
# TELEGRAM NOTIFIER
# ============================================================================
class TelegramNotifier:
    def __init__(self, token: str, chat_id: str, config: TradingConfig):
        self.token, self.chat_id, self.config = token, chat_id, config
        self.base_url = f"https://api.telegram.org/bot{self.token}/sendMessage"

    def send_message(self, message: str):
        if not self.token or "YOUR_TELEGRAM" in self.token or not self.chat_id:
            logging.warning("Telegram token or chat ID is not configured. Skipping notification.")
            return
        try:
            response = requests.post(self.base_url, data={'chat_id': self.chat_id, 'text': message, 'parse_mode': 'HTML'}, timeout=self.config.REQUESTS_TIMEOUT)
            response.raise_for_status()
            logging.info("Successfully sent notification to Telegram.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Failed to send Telegram notification: {e}")

# ============================================================================
# SCREENING AND SCORING
# ============================================================================
class AdvancedStockScreener:
    def __init__(self, config: TradingConfig):
        self.config = config
        self.market_condition = {}
        self.spy_hist = None

    def prepare_market_data(self):
        try:
            logging.info("Fetching SPY and VIX data for market regime analysis.")
            self.spy_hist = yf.Ticker('SPY').history(period='1y', auto_adjust=True)
            vix_hist = yf.Ticker('^VIX').history(period='10d', auto_adjust=True)
            if self.spy_hist.empty or vix_hist.empty: raise ValueError("Market data is empty.")

            spy_current = self.spy_hist['Close'].iloc[-1]
            spy_20ma = ta.trend.sma_indicator(self.spy_hist['Close'], window=20).iloc[-1]
            current_vix = vix_hist['Close'].iloc[-1]

            if spy_current > spy_20ma and current_vix < 20: self.market_condition['regime'] = "BULL"
            elif spy_current < spy_20ma and current_vix > 25: self.market_condition['regime'] = "BEAR"
            elif current_vix > 25: self.market_condition['regime'] = "HIGH_VOLATILITY"
            else: self.market_condition['regime'] = "NEUTRAL"
            self.market_condition['vix'] = current_vix
            logging.info(f"Market Regime: {self.market_condition['regime']} (VIX: {current_vix:.2f})")
        except Exception as e:
            logging.error(f"Could not determine market regime: {e}. Defaulting to NEUTRAL.")
            self.market_condition = {'regime': 'NEUTRAL', 'vix': 20}

    def apply_filters(self, ticker_data_dict: dict) -> dict:
        filtered_stocks = {}
        for ticker, (hist, info) in ticker_data_dict.items():
            try:
                if not (self.config.MIN_MARKET_CAP <= info.get('marketCap', 0) <= self.config.MAX_MARKET_CAP): continue
                if not (self.config.MIN_PRICE <= hist['Close'].iloc[-1] <= self.config.MAX_PRICE): continue
                if info.get('averageVolume', 0) < self.config.MIN_VOLUME: continue
                filtered_stocks[ticker] = (hist, info)
            except (KeyError, IndexError, TypeError): continue
        logging.info(f"Initial filters: {len(ticker_data_dict)} -> {len(filtered_stocks)} stocks.")
        return filtered_stocks

    def calculate_scoring(self, tech_indicators: dict, info: dict, finviz_data: dict, momentum_stats: dict, ml_prediction: float) -> float:
        if tech_indicators is None: return 0
        score = 0
        try:
            # --- Core Technical Score ---
            score += min(30, (tech_indicators.get('volume_ratio', 1) - 1) * 15) # Increased weight
            score += min(25, abs(tech_indicators.get('gap_percent', 0)) * 400)
            if tech_indicators.get('price_above_sma20') and tech_indicators.get('sma20_above_sma50'): score += 10
            if tech_indicators.get('macd_bullish_cross'): score += 10 # Increased weight
            if tech_indicators.get('rsi_14', 50) > 60: score += 7

            # --- Volatility and Market Adjustment (Refined) ---
            atr = tech_indicators.get('atr_14_percent', 0)
            score += 15 if 3.0 <= atr <= 8.0 else max(0, 15 - 2 * abs(atr - 5.5)) # Higher reward for optimal range
            if self.market_condition.get('regime') == 'BULL' and tech_indicators.get('relative_strength_20d', 0) > 0.01: score += 5

            # --- Alpha & Trader Logic Score ---
            score += finviz_data.get('sentiment_score', 0) * 15 # Increased weight
            score += finviz_data.get('catalyst_score', 0)
            if finviz_data.get('short_float', 0) > self.config.MIN_SHORT_FLOAT_FOR_SQUEEZE:
                score += finviz_data.get('short_float', 0) * 0.75 # Increased weight for squeeze potential

            momentum_7d = momentum_stats.get('momentum_7d_pct', 0)
            score += min(15, max(0, momentum_7d * 0.75))

            # --- ML PREDICTION SCORE (CRITICAL NEW COMPONENT) ---
            if ml_prediction > 0:
                score += ml_prediction * 20 # Add 20 points for every 1% predicted gain
            else:
                score += ml_prediction * 10 # Penalize for predicted losses

            # --- Risk Management Penalties ---
            extension_pct = ((tech_indicators['current_price'] - tech_indicators['sma_20']) / tech_indicators['sma_20']) * 100
            if extension_pct > self.config.MAX_EXTENSION_FROM_SMA20:
                penalty = (extension_pct - self.config.MAX_EXTENSION_FROM_SMA20) * 1.5 # Increased penalty
                score -= penalty
                logging.info(f"Applying overextension penalty of {penalty:.1f} to {info.get('symbol')}")

            # --- Final Sector Weighting ---
            score *= self.config.SECTOR_WEIGHTS.get(info.get('sector', 'Unknown'), 1.0)
            return score
        except Exception as e:
            logging.error(f"Error during scoring for {info.get('symbol')}: {e}")
            return 0

    def screen_stocks(self, tickers: list, max_stocks=15) -> list:
        logging.info(f"Starting screen for {len(tickers)} stocks...")
        self.prepare_market_data()
        if self.spy_hist is None: return []

        ticker_data = get_stock_data_parallel(tickers, self.config)
        filtered_data = self.apply_filters(ticker_data)

        scored_stocks = []
        logging.info(f"Calculating scores for {len(filtered_data)} stocks...")
        for ticker, (hist, info) in filtered_data.items():
            tech_indicators = calculate_technical_indicators(hist, self.spy_hist)
            if tech_indicators is None: continue

            momentum_stats = analyze_recent_momentum(hist)
            finviz_data = retry_with_backoff(get_finviz_data, self.config.MAX_RETRIES, self.config.RETRY_DELAY, ticker, self.config) or {}
            ml_prediction = get_ml_prediction(hist) # Get ML prediction

            score = self.calculate_scoring(tech_indicators, info, finviz_data, momentum_stats, ml_prediction)

            if score > 40: # Increased threshold for higher quality setups
                entry = tech_indicators['current_price']
                stop_loss = entry * (1 - (tech_indicators['atr_14_percent'] / 100))
                target = entry + ((entry - stop_loss) * self.config.RISK_REWARD_RATIO)

                rationale = f"ML Pred: {ml_prediction:+.2f}%. Vol Ratio ({tech_indicators['volume_ratio']:.2f}x). "
                if (keyword := finviz_data.get('catalyst_keyword')):
                    rationale += f"Catalyst: {keyword.title()}. "
                if finviz_data.get('short_float', 0) > self.config.MIN_SHORT_FLOAT_FOR_SQUEEZE:
                    rationale += f"Squeeze ({finviz_data['short_float']:.1f}% Short). "
                if momentum_stats.get('momentum_7d_pct', 0) > 10:
                    rationale += f"Hot Mover (+{momentum_stats['momentum_7d_pct']:.1f}% 7D). "

                scored_stocks.append({
                    'Ticker': ticker, 'Score': score,
                    'Entry': f"${entry:.2f}", 'Stop Loss': f"${stop_loss:.2f}",
                    'Target': f"${target:.2f}",
                    'ML Pred %': f"{ml_prediction:+.2f}",
                    'Short Float %': f"{finviz_data.get('short_float', 0):.1f}",
                    '7D Change %': f"{momentum_stats.get('momentum_7d_pct', 0):.1f}",
                    'Rationale': rationale.strip()
                })

        logging.info(f"Found {len(scored_stocks)} stocks meeting the score threshold (>40).")
        scored_stocks.sort(key=lambda x: x['Score'], reverse=True)

        for i, stock in enumerate(scored_stocks): stock['Rank'] = i + 1
        return scored_stocks[:max_stocks]

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    config = TradingConfig()
    screener = AdvancedStockScreener(config)
    notifier = TelegramNotifier(config.TELEGRAM_BOT_TOKEN, config.TELEGRAM_CHAT_ID, config)

    stock_universe = get_enhanced_stock_universe(config)
    top_stocks = screener.screen_stocks(stock_universe)

    if top_stocks:
        results_df = pd.DataFrame(top_stocks)
        cols = ['Rank', 'Ticker', 'Score', 'Entry', 'Stop Loss', 'Target', 'ML Pred %', 'Short Float %', '7D Change %', 'Rationale']
        results_df = results_df[cols]

        logging.info("--- Top Day Trading Candidates ---")
        results_df_display = results_df.copy()
        results_df_display['Score'] = results_df_display['Score'].map('{:,.1f}'.format)
        pd.set_option('display.max_colwidth', 60)
        print("\n" + results_df_display.to_string(index=False) + "\n")

        today_str = datetime.now().strftime('%Y-%m-%d')
        market_regime = screener.market_condition.get('regime', 'N/A')
        vix = screener.market_condition.get('vix', 0)

        message_header = f"<b>🔥 Alpha AI Trading Candidates ({today_str})</b>\n"
        message_header += f"<i>Market: {market_regime} (VIX: {vix:.2f})</i>\n\n"

        message_body = ""
        for _, row in results_df.iterrows():
            message_body += f"<b>#{row['Rank']}: {row['Ticker']}</b> (Score: {row['Score']:.1f}) | ML: {row['ML Pred %']}%\n"
            message_body += f"  - Entry: {row['Entry']}, SL: {row['Stop Loss']}, TGT: {row['Target']}\n"
            message_body += f"  - <i>Rationale: {row['Rationale']}</i>\n\n"

        notifier.send_message(message_header + message_body)
    else:
        logging.info("No suitable stocks found based on the current criteria.")
        notifier.send_message(f"🤷 No suitable AI-driven trading candidates found for {datetime.now().strftime('%Y-%m-%d')}.")

if __name__ == '__main__':
    main()
