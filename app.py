# =============================================================================
# app.py - Исправленная версия с диагностикой модели
# =============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime
import traceback

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Глобальная переменная для модели
model_predictor = None
model_load_error = None

class SolanaTokenPredictor:
    """Класс для предсказания токенов Solana"""
    
    def __init__(self):
        self.model = None
        self.imputer = None
        self.feature_names = None
        self.feature_importance = None
        self.model_metadata = {}
        self.is_trained = False
        
    def clean_numeric_column(self, series):
        """Очищает столбец от нечисловых символов и конвертирует в числа"""
        if series.dtype == 'object':
            series = series.astype(str)
            series = series.str.replace(r'[$,%\s]', '', regex=True)
            series = series.str.replace('K', 'e3', regex=False)
            series = series.str.replace('M', 'e6', regex=False)
            series = series.str.replace('B', 'e9', regex=False)
            series = series.str.replace('T', 'e12', regex=False)
            series = pd.to_numeric(series, errors='coerce')
        return series
    
    def prepare_features(self, df):
        """Подготавливает признаки из входных данных"""
        df = df.copy()
        
        # Обработка token_age
        if 'token_age' in df.columns:
            df['token_age_minutes'] = df['token_age'].str.extract(r'(\d+)').astype(float)
        
        # Очистка числовых столбцов
        numeric_columns_to_clean = ['market_cap', 'liquidity', 'ath']
        for col in numeric_columns_to_clean:
            if col in df.columns:
                df[col] = self.clean_numeric_column(df[col])
        
        # Извлечение признаков из nested объектов
        if 'first_buyers' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row['first_buyers']) and isinstance(row['first_buyers'], dict):
                    buyers = row['first_buyers']
                    df.loc[idx, 'buyers_green'] = buyers.get('green', 0)
                    df.loc[idx, 'buyers_blue'] = buyers.get('blue', 0)
                    df.loc[idx, 'buyers_yellow'] = buyers.get('yellow', 0)
                    df.loc[idx, 'buyers_red'] = buyers.get('red', 0)
                    df.loc[idx, 'buyers_clown'] = buyers.get('clown', 0)
                    df.loc[idx, 'buyers_sun'] = buyers.get('sun', 0)
                    df.loc[idx, 'buyers_moon_half'] = buyers.get('moon_half', 0)
                    df.loc[idx, 'buyers_moon_new'] = buyers.get('moon_new', 0)
        
        if 'current_initial_ratio' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row['current_initial_ratio']) and isinstance(row['current_initial_ratio'], dict):
                    ratio = row['current_initial_ratio']
                    df.loc[idx, 'current_ratio'] = ratio.get('current', 0)
                    df.loc[idx, 'initial_ratio'] = ratio.get('initial', 0)
        
        if 'security' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row['security']) and isinstance(row['security'], dict):
                    security = row['security']
                    df.loc[idx, 'security_no_mint'] = int(security.get('no_mint', False))
                    df.loc[idx, 'security_blacklist'] = int(security.get('blacklist', False))
                    df.loc[idx, 'security_burnt'] = int(security.get('burnt', False))
                    df.loc[idx, 'security_dev_sold'] = int(security.get('dev_sold', False))
                    df.loc[idx, 'security_dex_paid'] = int(security.get('dex_paid', False))
        
        if 'top_10_holdings' in df.columns:
            for idx, row in df.iterrows():
                if pd.notna(row['top_10_holdings']) and isinstance(row['top_10_holdings'], list):
                    df.loc[idx, 'top_10_holdings'] = sum(row['top_10_holdings'])
        
        # Базовые признаки
        base_features = [
            'market_cap', 'liquidity', 'sol_pooled', 'ath',
            'volume_1m', 'buy_volume_1m', 'sell_volume_1m', 'buys_1m', 'sells_1m',
            'volume_5m', 'buy_volume_5m', 'sell_volume_5m', 'buys_5m', 'sells_5m',
            'buyers_green', 'buyers_blue', 'buyers_yellow', 'buyers_red',
            'buyers_clown', 'buyers_sun', 'buyers_moon_half', 'buyers_moon_new',
            'current_ratio', 'initial_ratio', 'total_holders',
            'freshies_1d_percent', 'freshies_7d_percent', 'top_10_percent',
            'top_10_holdings', 'dev_current_balance_percent', 'dev_sol_balance',
            'token_age_minutes',
            'security_no_mint', 'security_blacklist', 'security_burnt',
            'security_dev_sold', 'security_dex_paid'
        ]
        
        available_features = [col for col in base_features if col in df.columns]
        
        # Feature Engineering
        engineered_features = []
        
        # Отношения объемов
        if 'buy_volume_1m' in df.columns and 'sell_volume_1m' in df.columns:
            df['buy_sell_ratio_1m'] = (df['buy_volume_1m'] + 1) / (df['sell_volume_1m'] + 1)
            engineered_features.append('buy_sell_ratio_1m')
        
        if 'volume_1m' in df.columns and 'volume_5m' in df.columns:
            df['volume_acceleration'] = (df['volume_1m'] + 1) / (df['volume_5m'] + 1)
            engineered_features.append('volume_acceleration')
        
        # Логарифмические трансформации
        log_features = ['market_cap', 'liquidity', 'volume_1m', 'volume_5m', 'total_holders']
        for feature in log_features:
            if feature in df.columns:
                df[f'log_{feature}'] = np.log1p(df[feature].fillna(0))
                engineered_features.append(f'log_{feature}')
        
        # Суммарные метрики
        buyer_features = ['buyers_green', 'buyers_blue', 'buyers_yellow', 'buyers_red']
        sniper_features = ['buyers_clown', 'buyers_sun', 'buyers_moon_half', 'buyers_moon_new']
        
        available_buyer_features = [f for f in buyer_features if f in df.columns]
        available_sniper_features = [f for f in sniper_features if f in df.columns]
        
        if len(available_buyer_features) > 1:
            df['total_regular_buyers'] = df[available_buyer_features].sum(axis=1)
            engineered_features.append('total_regular_buyers')
        
        if len(available_sniper_features) > 1:
            df['total_snipers'] = df[available_sniper_features].sum(axis=1)
            engineered_features.append('total_snipers')
        
        # Позитивные vs негативные сигналы
        positive_features = ['buyers_green', 'buyers_blue', 'buyers_clown', 'buyers_sun']
        negative_features = ['buyers_red', 'buyers_moon_new']
        
        available_positive = [f for f in positive_features if f in df.columns]
        available_negative = [f for f in negative_features if f in df.columns]
        
        if len(available_positive) > 0:
            df['positive_signals'] = df[available_positive].sum(axis=1)
            engineered_features.append('positive_signals')
        
        if len(available_negative) > 0:
            df['negative_signals'] = df[available_negative].sum(axis=1)
            engineered_features.append('negative_signals')
        
        if 'positive_signals' in engineered_features and 'negative_signals' in engineered_features:
            df['signal_ratio'] = (df['positive_signals'] + 1) / (df['negative_signals'] + 1)
            engineered_features.append('signal_ratio')
        
        # Показатели безопасности
        security_features = ['security_no_mint', 'security_blacklist', 'security_burnt', 
                           'security_dev_sold', 'security_dex_paid']
        available_security = [f for f in security_features if f in df.columns]
        
        if len(available_security) > 1:
            df['security_score'] = df[available_security].sum(axis=1)
            engineered_features.append('security_score')
        
        # Возрастные группы токенов
        if 'token_age_minutes' in df.columns:
            df['token_age_hours'] = df['token_age_minutes'] / 60
            df['is_fresh_token'] = (df['token_age_minutes'] <= 60).astype(int)
            df['is_mature_token'] = (df['token_age_minutes'] >= 1440).astype(int)
            engineered_features.extend(['token_age_hours', 'is_fresh_token', 'is_mature_token'])
        
        # Коэффициент активности
        if 'volume_1m' in df.columns and 'liquidity' in df.columns:
            df['volume_to_liquidity'] = np.log1p(df['volume_1m']) / np.log1p(df['liquidity'] + 1)
            engineered_features.append('volume_to_liquidity')
        
        final_features = available_features + engineered_features
        return df, final_features
    
    def predict(self, token_data, return_probability=True, return_confidence=False):
        """Предсказывает успешность токена"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # Подготавливаем данные
        if isinstance(token_data, dict):
            df_new = pd.DataFrame([token_data])
        elif isinstance(token_data, list):
            df_new = pd.DataFrame(token_data)
        else:
            df_new = token_data.copy()
        
        # Применяем feature engineering
        df_processed, _ = self.prepare_features(df_new)
        
        # Добавляем недостающие столбцы
        for col in self.feature_names:
            if col not in df_processed.columns:
                df_processed[col] = 0
        
        # Упорядочиваем столбцы
        df_processed = df_processed[self.feature_names]
        
        # Применяем импутер
        df_imputed = pd.DataFrame(
            self.imputer.transform(df_processed), 
            columns=self.feature_names
        )
        
        # Получаем предсказания
        predictions = self.model.predict(df_imputed)
        probabilities = self.model.predict_proba(df_imputed)[:, 1]
        
        results = []
        for i, (prediction, probability) in enumerate(zip(predictions, probabilities)):
            result = {
                'prediction': 'SUCCESS' if prediction == 1 else 'FAIL',
                'binary_prediction': int(prediction),
                'expected_pnl_category': f'PNL >= {self.model_metadata["target_threshold"]}x' if prediction == 1 else f'PNL < {self.model_metadata["target_threshold"]}x'
            }
            
            if return_probability:
                result['success_probability'] = round(float(probability), 4)
                result['success_probability_percent'] = f"{probability*100:.1f}%"
            
            if return_confidence:
                confidence_score = abs(probability - 0.5) * 2
                if confidence_score > 0.8:
                    confidence_level = "Very High"
                elif confidence_score > 0.6:
                    confidence_level = "High"
                elif confidence_score > 0.4:
                    confidence_level = "Medium"
                else:
                    confidence_level = "Low"
                
                result['confidence_score'] = round(float(confidence_score), 4)
                result['confidence_level'] = confidence_level
            
            results.append(result)
        
        return results[0] if len(results) == 1 else results
    
    def load_model(self, filepath):
        """Загружает сохраненную модель"""
        try:
            logger.info(f"Attempting to load model from: {filepath}")
            
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")
            
            file_size = os.path.getsize(filepath) / (1024 * 1024)
            logger.info(f"Model file size: {file_size:.1f}MB")
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            logger.info("Model data loaded successfully")
            
            # Проверяем структуру данных
            required_keys = ['model', 'imputer', 'feature_names', 'is_trained']
            missing_keys = [key for key in required_keys if key not in model_data]
            
            if missing_keys:
                raise ValueError(f"Missing required keys in model data: {missing_keys}")
            
            self.model = model_data['model']
            self.imputer = model_data['imputer']
            self.feature_names = model_data['feature_names']
            self.feature_importance = model_data.get('feature_importance', None)
            self.model_metadata = model_data.get('model_metadata', {})
            self.is_trained = model_data['is_trained']
            
            logger.info(f"Model loaded successfully!")
            logger.info(f"Features count: {len(self.feature_names)}")
            logger.info(f"Model metadata: {self.model_metadata}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            logger.error(traceback.format_exc())
            raise e

def find_model_file():
    """Ищет файл модели в различных возможных местах"""
    possible_paths = [
        "solana_model.pkl",
        "./solana_model.pkl",
        "solana_model_v2.pkl", 
        "./solana_model_v2.pkl",
        "model.pkl",
        "./model.pkl",
        os.path.join(os.path.dirname(__file__), 'solana_model.pkl'),
        os.path.join(os.path.dirname(__file__), 'solana_model_v2.pkl')
    ]
    
    logger.info("Searching for model file...")
    logger.info(f"Current directory: {os.getcwd()}")
    logger.info(f"Files in directory: {os.listdir('.')}")
    
    for path in possible_paths:
        if os.path.exists(path):
            logger.info(f"Found model file: {path}")
            return path
    
    logger.error("No model file found in any of the expected locations")
    return None

def load_model():
    """Загружает модель при запуске приложения"""
    global model_predictor, model_load_error
    
    try:
        logger.info("Starting model loading process...")
        
        model_predictor = SolanaTokenPredictor()
        
        # Ищем файл модели
        model_path = find_model_file()
        
        if model_path:
            model_predictor.load_model(model_path)
            logger.info("✅ Model loaded successfully on startup")
        else:
            model_load_error = "Model file not found"
            logger.error("❌ Model file not found")
            
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"❌ Failed to load model on startup: {str(e)}")
        logger.error(traceback.format_exc())

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Solana Token Predictor API',
        'version': '1.0.0',
        'model_loaded': model_predictor is not None and model_predictor.is_trained,
        'model_error': model_load_error if model_load_error else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/debug', methods=['GET'])
def debug_info():
    """Debug endpoint для диагностики"""
    current_dir = os.getcwd()
    files_in_dir = os.listdir('.')
    
    return jsonify({
        'current_directory': current_dir,
        'files_in_directory': files_in_dir,
        'model_loaded': model_predictor is not None and model_predictor.is_trained,
        'model_error': model_load_error,
        'model_metadata': model_predictor.model_metadata if model_predictor and model_predictor.is_trained else None,
        'feature_count': len(model_predictor.feature_names) if model_predictor and model_predictor.feature_names else 0
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Основной endpoint для предсказаний"""
    try:
        # Проверяем, что модель загружена
        if model_predictor is None or not model_predictor.is_trained:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The prediction model is not available',
                'details': model_load_error if model_load_error else 'Unknown error'
            }), 500
        
        # Получаем данные
        data = request.get_json()
        
        if not data:
            return jsonify({
                'error': 'No data provided',
                'message': 'Please provide token data in JSON format'
            }), 400
        
        # Выполняем предсказание
        result = model_predictor.predict(data, return_confidence=True)
        
        # Формируем ответ
        response = {
            'success': True,
            'data': result,
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'version': model_predictor.model_metadata.get('training_date', 'unknown'),
                'target_threshold': model_predictor.model_metadata.get('target_threshold', 2.0),
                'test_auc': model_predictor.model_metadata.get('test_auc', 'unknown')
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        return jsonify({
            'error': 'Prediction failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        if model_predictor is None or not model_predictor.is_trained:
            return jsonify({
                'error': 'Model not loaded',
                'message': 'The prediction model is not available'
            }), 500
        
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({
                'error': 'Invalid data format',
                'message': 'Please provide an array of token data'
            }), 400
        
        if len(data) > 100:
            return jsonify({
                'error': 'Too many requests',
                'message': 'Maximum 100 tokens per batch request'
            }), 400
        
        # Выполняем предсказания
        results = []
        for i, token_data in enumerate(data):
            try:
                result = model_predictor.predict(token_data, return_confidence=True)
                result['index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'index': i,
                    'error': str(e),
                    'prediction': 'ERROR'
                })
        
        response = {
            'success': True,
            'data': results,
            'total_processed': len(results),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        
        return jsonify({
            'error': 'Batch prediction failed',
            'message': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Информация о модели"""
    try:
        if model_predictor is None or not model_predictor.is_trained:
            return jsonify({
                'error': 'Model not loaded',
                'details': model_load_error if model_load_error else 'Unknown error'
            }), 500
        
        # Топ-10 важных признаков
        top_features = []
        if model_predictor.feature_importance is not None:
            top_10 = model_predictor.feature_importance.head(10)
            top_features = [
                {
                    'feature': row['feature'],
                    'importance': float(row['importance'])
                }
                for _, row in top_10.iterrows()
            ]
        
        response = {
            'model_metadata': model_predictor.model_metadata,
            'feature_count': len(model_predictor.feature_names) if model_predictor.feature_names else 0,
            'top_features': top_features,
            'is_trained': model_predictor.is_trained
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': 'Failed to get model info',
            'message': str(e)
        }), 500

@app.route('/reload', methods=['POST'])
def reload_model():
    """Endpoint для перезагрузки модели"""
    try:
        global model_predictor, model_load_error
        
        logger.info("Manual model reload requested")
        model_load_error = None
        
        load_model()
        
        if model_predictor and model_predictor.is_trained:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully',
                'model_info': {
                    'feature_count': len(model_predictor.feature_names),
                    'metadata': model_predictor.model_metadata
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to reload model',
                'error': model_load_error
            }), 500
            
    except Exception as e:
        logger.error(f"Model reload error: {str(e)}")
        return jsonify({
            'success': False,
            'message': 'Model reload failed',
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'message': 'The requested endpoint does not exist'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'message': 'An unexpected error occurred'
    }), 500

if __name__ == '__main__':
    # Загружаем модель при запуске
    load_model()
    
    # Запускаем приложение
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
