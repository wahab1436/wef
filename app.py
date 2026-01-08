"""
COMPLETE Weather Anomaly Detection Dashboard - Single File
Full backend + frontend integration
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import xgboost as xgb
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import warnings
import joblib
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
import traceback
import json

# ============================================
# 1. CONFIGURATION
# ============================================

class Config:
    RAW_DATA_PATH = "data/raw/weather_alerts_raw.csv"
    PROCESSED_DATA_PATH = "data/processed/weather_alerts_processed.csv"
    AGGREGATED_DATA_PATH = "data/processed/weather_alerts_aggregated.csv"
    ANOMALY_OUTPUT_PATH = "data/output/anomaly_results.csv"
    FORECAST_OUTPUT_PATH = "data/output/forecast_results.csv"
    ANOMALY_MODEL_PATH = "models/isolation_forest.pkl"
    FORECAST_MODEL_PATH = "models/xgboost_forecast.pkl"
    
    BASE_URL = "https://www.weather.gov"
    ALERTS_URL = "https://www.weather.gov/alerts"
    FORECAST_URL = "https://www.weather.gov/wrh/TextProduct"
    USER_AGENT = "WeatherAnomalyDetection/1.0"
    REQUEST_TIMEOUT = 30
    MAX_RETRIES = 3
    SCRAPING_INTERVAL = 3600
    
    STOPWORDS_LANGUAGE = "english"
    ANOMALY_CONTAMINATION = 0.05
    FORECAST_HORIZON = 7
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    DASHBOARD_PORT = 8501
    DASHBOARD_HOST = "0.0.0.0"
    CACHE_TTL = 300
    
    ALERT_TYPES = {
        'flood': ['flood', 'flash flood', 'flooding'],
        'storm': ['thunderstorm', 'storm', 'severe storm', 'tornado'],
        'winter': ['winter', 'snow', 'ice', 'blizzard', 'freezing'],
        'fire': ['fire', 'wildfire', 'red flag'],
        'wind': ['wind', 'high wind', 'wind advisory'],
        'heat': ['heat', 'excessive heat', 'heat advisory'],
        'cold': ['cold', 'freeze', 'frost', 'wind chill'],
        'coastal': ['coastal', 'surf', 'tsunami'],
        'air': ['air quality', 'smoke', 'dust'],
        'marine': ['marine', 'small craft', 'gale']
    }
    
    REGIONS = [
        'northeast', 'southeast', 'midwest', 'south', 'west',
        'northwest', 'southwest', 'central', 'eastern', 'western',
        'northern', 'southern'
    ]
    
    SEVERITY_KEYWORDS = {
        'warning': ['warning', 'emergency', 'dangerous', 'severe'],
        'watch': ['watch', 'possible', 'potential'],
        'advisory': ['advisory', 'caution', 'alert']
    }

# ============================================
# 2. UTILITY FUNCTIONS
# ============================================

def setup_directories():
    """Create all required directories"""
    directories = [
        'data/raw',
        'data/processed',
        'data/output',
        'models',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    return True

def setup_logging():
    """Setup basic logging"""
    import logging
    os.makedirs('logs', exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/app.log'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# ============================================
# 3. WEB SCRAPER (BACKEND)
# ============================================

class WeatherAlertScraper:
    """Scrape weather alerts from weather.gov"""
    
    def __init__(self):
        self.base_url = Config.BASE_URL
        self.alerts_url = Config.ALERTS_URL
        self.forecast_url = Config.FORECAST_URL
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': Config.USER_AGENT,
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8'
        })
        self.retry_count = 0
    
    def make_request(self, url: str):
        """Make HTTP request with retry logic"""
        try:
            response = self.session.get(url, timeout=Config.REQUEST_TIMEOUT)
            response.raise_for_status()
            self.retry_count = 0
            return response
        except Exception as e:
            self.retry_count += 1
            if self.retry_count <= Config.MAX_RETRIES:
                time.sleep(2 ** self.retry_count)
                return self.make_request(url)
            else:
                return None
    
    def scrape_alerts(self):
        """Scrape current weather alerts"""
        response = self.make_request(self.alerts_url)
        if not response:
            return []
        
        try:
            soup = BeautifulSoup(response.content, 'html.parser')
            alerts = []
            
            # Look for alert containers
            alert_selectors = [
                'div.alert-item',
                'div.alertentry',
                'div.alert',
                'article.alert'
            ]
            
            for selector in alert_selectors:
                alert_containers = soup.select(selector)
                if alert_containers:
                    for container in alert_containers:
                        alert_data = self._parse_alert_container(container)
                        if alert_data:
                            alerts.append(alert_data)
                    break
            
            # Fallback
            if not alerts:
                all_divs = soup.find_all('div')
                for div in all_divs:
                    text = div.get_text().lower()
                    if any(keyword in text for keyword in ['warning', 'watch', 'advisory', 'alert']):
                        alert_data = self._parse_generic_alert(div)
                        if alert_data:
                            alerts.append(alert_data)
            
            return alerts
            
        except Exception as e:
            return []
    
    def _parse_alert_container(self, container):
        """Parse individual alert container"""
        try:
            title_elem = container.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b'])
            title = title_elem.get_text(strip=True) if title_elem else "Weather Alert"
            
            desc_elem = container.find(['p', 'div', 'span'])
            description = desc_elem.get_text(strip=True) if desc_elem else ""
            
            if len(description) < 20:
                description = container.get_text(strip=True)
                if title in description:
                    description = description.replace(title, '').strip()
            
            region = self._extract_region(title + " " + description)
            alert_type = self._classify_alert_type(title, description)
            severity = self._extract_severity(title)
            timestamp = datetime.utcnow()
            
            return {
                'alert_id': f"{timestamp.strftime('%Y%m%d%H%M%S')}_{hash(title) % 10000:04d}",
                'timestamp': timestamp.isoformat(),
                'title': title[:500],
                'description': description[:2000],
                'region': region,
                'alert_type': alert_type,
                'severity': severity,
                'source': 'weather.gov',
                'scraped_at': datetime.utcnow().isoformat()
            }
            
        except Exception:
            return None
    
    def _parse_generic_alert(self, element):
        """Parse generic alert element"""
        try:
            text = element.get_text(strip=True)
            if len(text) < 50:
                return None
            
            lines = text.split('\n')
            title = lines[0][:200] if lines else "Weather Alert"
            description = text[:1500]
            
            region = self._extract_region(text)
            alert_type = self._classify_alert_type(title, description)
            severity = self._extract_severity(title)
            
            return {
                'alert_id': f"generic_{hash(text) % 1000000:06d}",
                'timestamp': datetime.utcnow().isoformat(),
                'title': title,
                'description': description,
                'region': region,
                'alert_type': alert_type,
                'severity': severity,
                'source': 'weather.gov',
                'scraped_at': datetime.utcnow().isoformat()
            }
        except Exception:
            return None
    
    def _extract_region(self, text: str):
        """Extract region from text"""
        text_lower = text.lower()
        
        for region in Config.REGIONS:
            if region in text_lower:
                return region.capitalize()
        
        states = ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA','HI','ID','IL','IN','IA',
                  'KS','KY','LA','ME','MD','MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
                  'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC','SD','TN','TX','UT','VT',
                  'VA','WA','WV','WI','WY']
        
        for state in states:
            if f" {state} " in f" {text} ":
                return state
        
        return "National"
    
    def _classify_alert_type(self, title: str, description: str):
        """Classify alert type based on keywords"""
        text = f"{title} {description}".lower()
        
        for alert_type, keywords in Config.ALERT_TYPES.items():
            if any(keyword in text for keyword in keywords):
                return alert_type
        
        return 'other'
    
    def _extract_severity(self, title: str):
        """Extract severity level from title"""
        title_lower = title.lower()
        
        for severity, keywords in Config.SEVERITY_KEYWORDS.items():
            if any(keyword in title_lower for keyword in keywords):
                return severity
        
        return 'unknown'
    
    def run_scraping_job(self, raw_data_path: str = None):
        """Run complete scraping job"""
        if raw_data_path is None:
            raw_data_path = Config.RAW_DATA_PATH
        
        alerts = self.scrape_alerts()
        
        if alerts:
            new_df = pd.DataFrame(alerts)
            
            if os.path.exists(raw_data_path):
                existing_df = pd.read_csv(raw_data_path)
                combined = pd.concat([existing_df, new_df])
                combined = combined.drop_duplicates(
                    subset=['title', 'timestamp'], 
                    keep='last'
                )
            else:
                combined = new_df
            
            os.makedirs(os.path.dirname(raw_data_path), exist_ok=True)
            combined.to_csv(raw_data_path, index=False)
            
            return len(alerts)
        else:
            return 0

# ============================================
# 4. TEXT PREPROCESSOR (BACKEND)
# ============================================

class TextPreprocessor:
    """Preprocess and extract features from text data"""
    
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except:
            nltk.download('punkt')
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words(Config.STOPWORDS_LANGUAGE))
    
    def clean_text(self, text: str):
        """Clean and normalize text"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s°\-%]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def extract_weather_features(self, text: str):
        """Extract weather-specific features from text"""
        text_lower = text.lower()
        features = {}
        
        for keyword in ['flood', 'storm', 'rain', 'snow', 'ice', 'wind', 'heat', 'cold', 'fire']:
            features[f'contains_{keyword}'] = int(keyword in text_lower)
        
        severity_indicators = {
            'warning': ['warning', 'emergency', 'dangerous', 'severe'],
            'caution': ['advisory', 'watch', 'caution', 'alert'],
            'information': ['statement', 'update', 'information']
        }
        
        for severity, indicators in severity_indicators.items():
            features[f'severity_{severity}'] = int(
                any(indicator in text_lower for indicator in indicators)
            )
        
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        
        return features
    
    def preprocess_dataframe(self, df: pd.DataFrame):
        """Preprocess entire dataframe"""
        if df.empty:
            return df
        
        df_processed = df.copy()
        
        if 'timestamp' in df_processed.columns:
            df_processed['timestamp'] = pd.to_datetime(df_processed['timestamp'])
            df_processed['date'] = df_processed['timestamp'].dt.date
            df_processed['hour'] = df_processed['timestamp'].dt.hour
            df_processed['day_of_week'] = df_processed['timestamp'].dt.dayofweek
        
        text_columns = ['title', 'description']
        for col in text_columns:
            if col in df_processed.columns:
                df_processed[f'{col}_cleaned'] = df_processed[col].apply(self.clean_text)
                
                feature_dicts = []
                for text in df_processed[f'{col}_cleaned']:
                    feature_dicts.append(self.extract_weather_features(text))
                
                feature_df = pd.DataFrame(feature_dicts)
                feature_df.columns = [f'{col}_{c}' for c in feature_df.columns]
                df_processed = pd.concat([df_processed, feature_df], axis=1)
        
        return df_processed
    
    def create_aggregated_dataset(self, df: pd.DataFrame):
        """Create aggregated dataset for ML models"""
        if df.empty:
            return pd.DataFrame()
        
        if 'date' not in df.columns and 'timestamp' in df.columns:
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
        
        if 'date' not in df.columns:
            return pd.DataFrame()
        
        df['date'] = pd.to_datetime(df['date'])
        aggregated = pd.DataFrame(index=pd.date_range(
            start=df['date'].min(),
            end=df['date'].max(),
            freq='D'
        ))
        aggregated.index.name = 'date'
        
        if 'alert_type' in df.columns:
            type_counts = df.groupby(['date', 'alert_type']).size().unstack(fill_value=0)
            aggregated = aggregated.join(type_counts)
        
        severity_cols = [col for col in df.columns if col.startswith('severity_')]
        if severity_cols:
            severity_counts = df.groupby('date')[severity_cols].sum()
            aggregated = aggregated.join(severity_counts)
        
        keyword_cols = [col for col in df.columns if col.startswith('contains_')]
        if keyword_cols:
            keyword_sums = df.groupby('date')[keyword_cols].sum()
            aggregated = aggregated.join(keyword_sums)
        
        aggregated['total_alerts'] = df.groupby('date').size()
        aggregated = aggregated.fillna(0)
        
        aggregated['day_of_week'] = aggregated.index.dayofweek
        aggregated['month'] = aggregated.index.month
        
        return aggregated

# ============================================
# 5. ANOMALY DETECTOR (BACKEND)
# ============================================

class AnomalyDetector:
    """Detect anomalies in weather alert patterns"""
    
    def __init__(self, contamination: float = None):
        self.contamination = contamination or Config.ANOMALY_CONTAMINATION
        self.model = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=Config.RANDOM_STATE,
            n_jobs=-1
        )
        self.scaler = RobustScaler()
        self.is_fitted = False
    
    def detect_anomalies(self, df: pd.DataFrame):
        """Detect anomalies in the dataset"""
        if df.empty:
            result_df = df.copy()
            result_df['is_anomaly'] = False
            result_df['anomaly_score'] = 0
            result_df['anomaly_probability'] = 0
            return result_df
        
        result_df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        exclude_patterns = ['date', 'day_of_week', 'month']
        feature_cols = [col for col in numeric_cols if not any(pattern in col for pattern in exclude_patterns)]
        
        if not feature_cols:
            result_df['is_anomaly'] = False
            result_df['anomaly_score'] = 0
            result_df['anomaly_probability'] = 0
            return result_df
        
        X = df[feature_cols].values
        X = np.nan_to_num(X)
        
        if self.is_fitted:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = self.scaler.fit_transform(X)
            self.is_fitted = True
        
        if not hasattr(self.model, 'estimators_'):
            self.model.fit(X_scaled)
        
        anomaly_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        result_df['is_anomaly'] = predictions == -1
        result_df['anomaly_score'] = anomaly_scores
        
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        if max_score > min_score:
            result_df['anomaly_probability'] = (anomaly_scores - min_score) / (max_score - min_score)
        else:
            result_df['anomaly_probability'] = 0.5
        
        return result_df

# ============================================
# 6. FORECAST MODEL (BACKEND)
# ============================================

class AlertForecaster:
    """Forecast future weather alerts"""
    
    def __init__(self, target_col: str = 'total_alerts', forecast_horizon: int = None):
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon or Config.FORECAST_HORIZON
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.is_fitted = False
        
        self.xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': Config.RANDOM_STATE,
            'n_jobs': -1,
            'verbosity': 0
        }
    
    def create_features(self, df: pd.DataFrame):
        """Create time series features for forecasting"""
        df_features = df.copy()
        
        # Create lag features
        for lag in [1, 2, 3, 7, 14]:
            if self.target_col in df_features.columns:
                df_features[f'{self.target_col}_lag_{lag}'] = df_features[self.target_col].shift(lag)
        
        # Create rolling statistics
        for window in [3, 7, 14]:
            if self.target_col in df_features.columns:
                df_features[f'{self.target_col}_rolling_mean_{window}'] = (
                    df_features[self.target_col].rolling(window=window).mean()
                )
                df_features[f'{self.target_col}_rolling_std_{window}'] = (
                    df_features[self.target_col].rolling(window=window).std()
                )
        
        # Create time-based features
        if hasattr(df_features.index, 'dayofweek'):
            df_features['day_of_week_sin'] = np.sin(2 * np.pi * df_features.index.dayofweek / 7)
            df_features['day_of_week_cos'] = np.cos(2 * np.pi * df_features.index.dayofweek / 7)
            df_features['month_sin'] = np.sin(2 * np.pi * df_features.index.month / 12)
            df_features['month_cos'] = np.cos(2 * np.pi * df_features.index.month / 12)
        
        df_features = df_features.dropna()
        return df_features
    
    def train(self, df: pd.DataFrame):
        """Train the forecasting model"""
        df_features = self.create_features(df)
        
        if self.target_col not in df_features.columns:
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                self.target_col = numeric_cols[0]
            else:
                return {}
        
        exclude_cols = [self.target_col]
        self.feature_columns = [
            col for col in df_features.columns 
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df_features[col])
        ]
        
        if not self.feature_columns:
            return {}
        
        X = df_features[self.feature_columns].values
        y = df_features[self.target_col].values
        
        X_scaled = self.scaler.fit_transform(X)
        
        split_idx = int(len(X) * (1 - Config.TEST_SIZE))
        if split_idx < 10:
            split_idx = max(10, len(X) - 5)
        
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        if len(X_train) < 10:
            return {}
        
        self.model = xgb.XGBRegressor(**self.xgb_params)
        self.model.fit(X_train, y_train, verbose=False)
        
        y_pred_test = self.model.predict(X_test)
        
        metrics = {
            'test_mae': mean_absolute_error(y_test, y_pred_test),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'test_r2': r2_score(y_test, y_pred_test),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
        
        self.is_fitted = True
        return metrics
    
    def forecast(self, df: pd.DataFrame, periods: int = None):
        """Generate forecasts for future periods"""
        if periods is None:
            periods = self.forecast_horizon
        
        if not self.is_fitted or self.model is None:
            return pd.DataFrame()
        
        df_forecast_base = df.copy()
        
        if hasattr(df_forecast_base.index, 'max'):
            last_date = df_forecast_base.index.max()
        else:
            last_date = datetime.now()
        
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        
        df_features = self.create_features(df_forecast_base)
        if df_features.empty:
            return pd.DataFrame()
        
        last_features = df_features.iloc[[-1]].copy()
        forecasts = []
        
        for i in range(periods):
            X_pred = last_features[self.feature_columns].values
            X_pred_scaled = self.scaler.transform(X_pred)
            
            pred = self.model.predict(X_pred_scaled)[0]
            forecasts.append(pred)
        
        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'forecast': forecasts,
            'lower_bound': np.array(forecasts) * 0.8,
            'upper_bound': np.array(forecasts) * 1.2
        })
        
        forecast_df['lower_bound'] = forecast_df['lower_bound'].clip(lower=0)
        forecast_df['upper_bound'] = forecast_df['upper_bound'].clip(lower=0)
        
        return forecast_df

# ============================================
# 7. DASHBOARD FRONTEND
# ============================================

class WeatherAnomalyDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_loaded = False
        self.df_alerts = None
        self.df_anomalies = None
        self.df_forecast = None
        self.last_updated = None
        self.setup_directories()
        self.load_data()
    
    def setup_directories(self):
        """Create required directories"""
        directories = [
            'data/raw',
            'data/processed',
            'data/output',
            'models',
            'logs'
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load all necessary data"""
        try:
            # Check if data files exist
            data_exists = (
                os.path.exists(Config.PROCESSED_DATA_PATH) and
                os.path.exists(Config.ANOMALY_OUTPUT_PATH) and
                os.path.exists(Config.FORECAST_OUTPUT_PATH)
            )
            
            if not data_exists:
                return False
            
            # Load processed alerts
            if os.path.exists(Config.PROCESSED_DATA_PATH):
                self.df_alerts = pd.read_csv(Config.PROCESSED_DATA_PATH)
                if 'timestamp' in self.df_alerts.columns:
                    self.df_alerts['timestamp'] = pd.to_datetime(self.df_alerts['timestamp'])
            else:
                self.df_alerts = pd.DataFrame()
            
            # Load anomaly results
            if os.path.exists(Config.ANOMALY_OUTPUT_PATH):
                self.df_anomalies = pd.read_csv(Config.ANOMALY_OUTPUT_PATH, index_col=0)
                if self.df_anomalies.index.name == 'date':
                    self.df_anomalies.index = pd.to_datetime(self.df_anomalies.index)
            else:
                self.df_anomalies = pd.DataFrame()
            
            # Load forecast results
            if os.path.exists(Config.FORECAST_OUTPUT_PATH):
                self.df_forecast = pd.read_csv(Config.FORECAST_OUTPUT_PATH)
                if 'date' in self.df_forecast.columns:
                    self.df_forecast['date'] = pd.to_datetime(self.df_forecast['date'])
            else:
                self.df_forecast = pd.DataFrame()
            
            self.data_loaded = True
            self.last_updated = datetime.now()
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return False
    
    def get_summary_metrics(self):
        """Calculate summary metrics"""
        metrics = {
            'total_alerts': 0,
            'recent_alerts': 0,
            'anomaly_count': 0,
            'top_region': 'N/A',
            'top_alert_type': 'N/A',
            'forecast_avg': 0
        }
        
        if not self.data_loaded:
            return metrics
        
        if not self.df_alerts.empty:
            metrics['total_alerts'] = len(self.df_alerts)
            
            if 'timestamp' in self.df_alerts.columns:
                recent_cutoff = datetime.now() - timedelta(days=7)
                recent_alerts = self.df_alerts[self.df_alerts['timestamp'] > recent_cutoff]
                metrics['recent_alerts'] = len(recent_alerts)
            
            if 'region' in self.df_alerts.columns:
                region_counts = self.df_alerts['region'].value_counts()
                if not region_counts.empty:
                    metrics['top_region'] = region_counts.index[0]
            
            if 'alert_type' in self.df_alerts.columns:
                type_counts = self.df_alerts['alert_type'].value_counts()
                if not type_counts.empty:
                    metrics['top_alert_type'] = type_counts.index[0]
        
        if not self.df_anomalies.empty and 'is_anomaly' in self.df_anomalies.columns:
            metrics['anomaly_count'] = int(self.df_anomalies['is_anomaly'].sum())
        
        if not self.df_forecast.empty and 'forecast' in self.df_forecast.columns:
            metrics['forecast_avg'] = float(self.df_forecast['forecast'].mean())
        
        return metrics
    
    def create_alert_trend_chart(self):
        """Create alert trend chart with anomalies"""
        if self.df_anomalies.empty or 'total_alerts' not in self.df_anomalies.columns:
            return None
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=self.df_anomalies.index,
            y=self.df_anomalies['total_alerts'],
            mode='lines',
            name='Daily Alerts',
            line=dict(color='#007bff', width=2)
        ))
        
        if 'is_anomaly' in self.df_anomalies.columns:
            anomalies = self.df_anomalies[self.df_anomalies['is_anomaly']]
            if not anomalies.empty and 'total_alerts' in anomalies.columns:
                fig.add_trace(go.Scatter(
                    x=anomalies.index,
                    y=anomalies['total_alerts'],
                    mode='markers',
                    name='Anomalies',
                    marker=dict(
                        color='#dc3545',
                        size=10,
                        symbol='diamond'
                    )
                ))
        
        if not self.df_forecast.empty:
            fig.add_trace(go.Scatter(
                x=self.df_forecast['date'],
                y=self.df_forecast['forecast'],
                mode='lines',
                name='Forecast',
                line=dict(color='#28a745', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Weather Alert Trends with Anomaly Detection',
            xaxis_title='Date',
            yaxis_title='Number of Alerts',
            height=500
        )
        
        return fig
    
    def create_alert_type_chart(self):
        """Create alert type distribution chart - FIXED VERSION"""
        if self.df_alerts.empty or 'alert_type' not in self.df_alerts.columns:
            return None
        
        # Create a proper DataFrame for Plotly
        alert_counts = self.df_alerts['alert_type'].value_counts().reset_index()
        alert_counts.columns = ['alert_type', 'count']
        
        fig = px.bar(
            alert_counts,
            x='alert_type',
            y='count',
            title='Alert Type Distribution',
            labels={'alert_type': 'Alert Type', 'count': 'Count'}
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_region_chart(self):
        """Create region distribution chart - FIXED VERSION"""
        if self.df_alerts.empty or 'region' not in self.df_alerts.columns:
            return None
        
        # Create a proper DataFrame for Plotly
        region_counts = self.df_alerts['region'].value_counts().reset_index()
        region_counts.columns = ['region', 'count']
        region_counts = region_counts.head(10)  # Top 10 regions
        
        fig = px.bar(
            region_counts,
            x='region',
            y='count',
            title='Top Regions by Alert Count',
            labels={'region': 'Region', 'count': 'Alert Count'}
        )
        
        fig.update_layout(height=400)
        return fig
    
    def create_severity_chart(self):
        """Create severity distribution chart"""
        if self.df_alerts.empty or 'severity' not in self.df_alerts.columns:
            return None
        
        severity_counts = self.df_alerts['severity'].value_counts().reset_index()
        severity_counts.columns = ['severity', 'count']
        
        fig = px.pie(
            severity_counts,
            values='count',
            names='severity',
            title='Alert Severity Distribution'
        )
        
        fig.update_layout(height=350)
        return fig
    
    def generate_insights(self):
        """Generate plain-English insights"""
        insights = []
        
        if not self.data_loaded:
            insights.append("Data is being loaded. Please wait.")
            return insights
        
        if self.df_alerts.empty:
            insights.append("No alert data available. Run the pipeline first.")
            return insights
        
        metrics = self.get_summary_metrics()
        
        if metrics['recent_alerts'] > 0:
            avg_daily = metrics['recent_alerts'] / 7
            insights.append(f"Average of {avg_daily:.1f} alerts per day in the last week.")
        
        if metrics['anomaly_count'] > 0:
            insights.append(f"Detected {metrics['anomaly_count']} unusual patterns requiring attention.")
        
        if metrics['top_alert_type'] != 'N/A':
            insights.append(f"Most frequent alert type: {metrics['top_alert_type']}.")
        
        if metrics['forecast_avg'] > 0:
            insights.append(f"Forecast predicts {metrics['forecast_avg']:.1f} alerts per day on average.")
        
        if metrics['top_region'] != 'N/A':
            insights.append(f"Highest alert activity in {metrics['top_region']} region.")
        
        return insights
    
    def run_backend_pipeline(self):
        """Run the complete backend pipeline"""
        try:
            # Step 1: Scrape data
            st.info("Step 1/4: Scraping weather alerts...")
            scraper = WeatherAlertScraper()
            count = scraper.run_scraping_job()
            st.success(f"Scraped {count} new alerts")
            
            # Step 2: Preprocess data
            st.info("Step 2/4: Processing data...")
            df_raw = pd.read_csv(Config.RAW_DATA_PATH)
            preprocessor = TextPreprocessor()
            df_processed = preprocessor.preprocess_dataframe(df_raw)
            df_aggregated = preprocessor.create_aggregated_dataset(df_processed)
            
            df_processed.to_csv(Config.PROCESSED_DATA_PATH, index=False)
            df_aggregated.to_csv(Config.AGGREGATED_DATA_PATH)
            st.success(f"Processed {len(df_processed)} alerts")
            
            # Step 3: Detect anomalies
            st.info("Step 3/4: Detecting anomalies...")
            detector = AnomalyDetector()
            df_anomalies = detector.detect_anomalies(df_aggregated)
            df_anomalies.to_csv(Config.ANOMALY_OUTPUT_PATH)
            
            anomaly_count = df_anomalies['is_anomaly'].sum() if 'is_anomaly' in df_anomalies.columns else 0
            st.success(f"Detected {anomaly_count} anomalies")
            
            # Step 4: Generate forecasts
            st.info("Step 4/4: Generating forecasts...")
            forecaster = AlertForecaster()
            metrics = forecaster.train(df_aggregated)
            forecasts = forecaster.forecast(df_aggregated)
            
            forecasts.to_csv(Config.FORECAST_OUTPUT_PATH, index=False)
            st.success(f"Generated {len(forecasts)} days of forecasts")
            
            # Save models
            joblib.dump(detector, Config.ANOMALY_MODEL_PATH)
            joblib.dump(forecaster, Config.FORECAST_MODEL_PATH)
            
            st.success("Pipeline completed successfully!")
            st.experimental_rerun()
            
            return True
            
        except Exception as e:
            st.error(f"Pipeline error: {str(e)}")
            return False
    
    def run(self):
        """Run the dashboard"""
        # Dashboard header
        st.title("Weather Anomaly Detection Dashboard")
        st.markdown("Real-time monitoring and forecasting of weather alert anomalies")
        
        if self.last_updated:
            st.caption(f"Last updated: {self.last_updated.strftime('%Y-%m-%d %H:%M:%S')}")
        
        st.markdown("---")
        
        # Control panel
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Run Full Pipeline", type="primary"):
                self.run_backend_pipeline()
        with col2:
            if st.button("Scrape New Data"):
                scraper = WeatherAlertScraper()
                count = scraper.run_scraping_job()
                st.success(f"Scraped {count} new alerts")
                st.experimental_rerun()
        with col3:
            if st.button("Refresh Dashboard"):
                st.experimental_rerun()
        
        st.markdown("---")
        
        # Check data status
        if not self.data_loaded:
            st.warning("No data available. Click 'Run Full Pipeline' to collect and process data.")
            
            with st.expander("Setup Instructions"):
                st.markdown("""
                1. **Click 'Run Full Pipeline'** to scrape, process, and analyze data
                2. **Wait for completion** (takes 1-2 minutes)
                3. **Dashboard will refresh automatically**
                
                **Data Sources:**
                - weather.gov/alerts
                - weather.gov/wrh/TextProduct
                
                **Pipeline Steps:**
                1. Scrape real-time weather alerts
                2. Clean and process text data
                3. Detect anomalies in alert patterns
                4. Generate 7-day forecasts
                """)
            
            return
        
        # Display metrics
        st.subheader("Summary Metrics")
        metrics = self.get_summary_metrics()
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Alerts", f"{metrics['total_alerts']:,}")
        with col2:
            st.metric("7-Day Alerts", f"{metrics['recent_alerts']:,}")
        with col3:
            st.metric("Anomalies", f"{metrics['anomaly_count']}")
        with col4:
            st.metric("Top Region", metrics['top_region'])
        with col5:
            st.metric("Forecast Avg", f"{metrics['forecast_avg']:.1f}")
        
        st.markdown("---")
        
        # Display insights
        st.subheader("Key Insights")
        insights = self.generate_insights()
        for insight in insights:
            st.info(insight)
        
        st.markdown("---")
        
        # Display charts
        col1, col2 = st.columns([2, 1])
        
        with col1:
            trend_chart = self.create_alert_trend_chart()
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            else:
                st.info("Trend chart data not available")
        
        with col2:
            severity_chart = self.create_severity_chart()
            if severity_chart:
                st.plotly_chart(severity_chart, use_container_width=True)
            else:
                st.info("Severity chart data not available")
        
        col3, col4 = st.columns(2)
        
        with col3:
            type_chart = self.create_alert_type_chart()
            if type_chart:
                st.plotly_chart(type_chart, use_container_width=True)
            else:
                st.info("Alert type chart data not available")
        
        with col4:
            region_chart = self.create_region_chart()
            if region_chart:
                st.plotly_chart(region_chart, use_container_width=True)
            else:
                st.info("Region chart data not available")
        
        st.markdown("---")
        
        # Data tables
        tab1, tab2, tab3 = st.tabs(["Recent Alerts", "Anomalies", "Forecast"])
        
        with tab1:
            if not self.df_alerts.empty:
                recent_alerts = self.df_alerts.sort_values('timestamp', ascending=False).head(20)
                st.dataframe(recent_alerts[['timestamp', 'title', 'region', 'alert_type', 'severity']], use_container_width=True)
            else:
                st.info("No alert data available")
        
        with tab2:
            if not self.df_anomalies.empty and 'is_anomaly' in self.df_anomalies.columns:
                anomalies = self.df_anomalies[self.df_anomalies['is_anomaly']].sort_index(ascending=False)
                st.dataframe(anomalies[['total_alerts', 'anomaly_score']].head(10), use_container_width=True)
            else:
                st.info("No anomaly data available")
        
        with tab3:
            if not self.df_forecast.empty:
                st.dataframe(self.df_forecast, use_container_width=True)
            else:
                st.info("No forecast data available")
        
        # Footer
        st.markdown("---")
        st.markdown("""
        **Data Source**: National Weather Service (weather.gov)  
        **Update Frequency**: Hourly  
        **Dashboard Version**: 1.0.0
        """)

# ============================================
# 8. MAIN APPLICATION
# ============================================

def main():
    """Main application function"""
    # Setup
    setup_directories()
    
    # Initialize dashboard
    dashboard = WeatherAnomalyDashboard()
    
    # Run dashboard
    dashboard.run()

# ============================================
# 9. STREAMLIT ENTRY POINT
# ============================================

if __name__ == "__main__":
    main()
