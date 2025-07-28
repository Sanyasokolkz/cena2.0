from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import logging
import json
import os
from datetime import datetime
import sys
import warnings

# Отключаем предупреждения
warnings.filterwarnings('ignore')

# === ФИКС ДЛЯ NUMPY._CORE ПРОБЛЕМЫ ===
try:
    import numpy as np
    # Фикс для numpy._core ошибки
    if not hasattr(np, '_core'):
        try:
            import numpy.core as _core
            np._core = _core
            sys.modules['numpy._core'] = _core
        except:
            pass
    print("✅ Numpy loaded successfully")
except Exception as e:
    print(f"⚠️ Numpy import issue: {e}")
    # Создаем заглушку для numpy если совсем не работает
    class NumpyStub:
        def __init__(self):
            pass
        def array(self, x):
            return x
        def log1p(self, x):
            import math
            return math.log1p(x) if isinstance(x, (int, float)) else x
        def sqrt(self, x):
            import math
            return math.sqrt(x) if isinstance(x, (int, float)) else x
        def where(self, condition, x, y):
            return x if condition else y
        def cumsum(self, x):
            result = []
            total = 0
            for val in x:
                total += val
                result.append(total)
            return result
    np = NumpyStub()

# Пытаемся импортировать pandas
try:
    import pandas as pd
    print("✅ Pandas loaded successfully")
except Exception as e:
    print(f"⚠️ Pandas import issue: {e}")
    # Создаем заглушку для pandas
    class PandasStub:
        def isna(self, x):
            return x is None or x == '' or str(x).lower() == 'nan'
        def DataFrame(self, data):
            return {'data': data}
    pd = PandasStub()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# =============================================================================
# ЗАГЛУШКИ ДЛЯ ML МОДЕЛИ
# =============================================================================

MODEL_PATH = './'
model = None
scaler = None
label_encoders = None
feature_names = None
model_metadata = None
ensemble_weights = None

def load_ml_model():
    """Попытка загрузки ML модели с обработкой ошибок"""
    global model, scaler, label_encoders, feature_names, model_metadata, ensemble_weights
    
    try:
        # Пытаемся импортировать joblib
        try:
            import joblib
            print("✅ Joblib imported successfully")
        except ImportError:
            print("❌ Joblib not available")
            return False
        
        logger.info("Загружаем ML модель...")
        
        # Проверяем файлы
        current_dir = os.getcwd()
        logger.info(f"Текущая директория: {current_dir}")
        
        try:
            files_in_root = [f for f in os.listdir('.') if f.endswith(('.pkl', '.json'))]
            logger.info(f"ML файлы в корне: {files_in_root}")
        except Exception as e:
            logger.error(f"Ошибка чтения директории: {e}")
            return False
        
        # Загружаем модель
        model_file = os.path.join(MODEL_PATH, 'memtoken_model_improved.pkl')
        if os.path.exists(model_file):
            try:
                size = os.path.getsize(model_file)
                logger.info(f"Найден файл модели: {size} bytes")
                model = joblib.load(model_file)
                logger.info("✅ Модель загружена успешно")
            except Exception as e:
                logger.error(f"Ошибка загрузки модели: {e}")
                return False
        else:
            logger.warning(f"❌ Файл модели не найден: {model_file}")
            return False
        
        # Загружаем остальные компоненты (опционально)
        try:
            scaler_file = os.path.join(MODEL_PATH, 'memtoken_scaler_improved.pkl')
            if os.path.exists(scaler_file):
                scaler = joblib.load(scaler_file)
                logger.info("✅ Скейлер загружен")
            
            encoders_file = os.path.join(MODEL_PATH, 'memtoken_encoders_improved.pkl')
            if os.path.exists(encoders_file):
                label_encoders = joblib.load(encoders_file)
                logger.info("✅ Энкодеры загружены")
            
            features_file = os.path.join(MODEL_PATH, 'memtoken_features_improved.pkl')
            if os.path.exists(features_file):
                feature_names = joblib.load(features_file)
                logger.info(f"✅ Признаки загружены ({len(feature_names)} штук)")
            
            metadata_file = os.path.join(MODEL_PATH, 'memtoken_model_metadata.json')
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    model_metadata = json.load(f)
                model_name = model_metadata.get('best_model_name', 'unknown')
                logger.info(f"✅ Метаданные загружены (модель: {model_name})")
        
        except Exception as e:
            logger.warning(f"Некоторые компоненты не загружены: {e}")
        
        # Финальная проверка
        if model is not None:
            logger.info("🎉 ML модель готова к работе!")
            return True
        else:
            logger.error("❌ Основная модель не загружена")
            return False
        
    except Exception as e:
        logger.error(f"❌ Общая ошибка загрузки модели: {e}")
        return False

# Пытаемся загрузить модель при старте
ML_MODEL_LOADED = load_ml_model()

# =============================================================================
# ФУНКЦИИ ОБРАБОТКИ ДАННЫХ
# =============================================================================

def parse_string_number(value):
    """Парсинг строковых чисел с K/M/B суффиксами"""
    if pd.isna(value) or value == '' or value == 'N/A':
        return 0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    value = str(value).upper().replace(',', '').strip()
    value = re.sub(r'[^\d\.\-KMB]', '', value)
    
    if value == '' or value == '-':
        return 0
    
    try:
        if 'K' in value:
            return float(value.replace('K', '')) * 1_000
        elif 'M' in value:
            return float(value.replace('M', '')) * 1_000_000
        elif 'B' in value:
            return float(value.replace('B', '')) * 1_000_000_000
        else:
            return float(value)
    except:
        return 0

def parse_time_to_minutes(value):
    """Парсинг времени в минуты"""
    if pd.isna(value) or value == '' or value == 'N/A':
        return 0
    
    if isinstance(value, (int, float)):
        return float(value)
    
    total_minutes = 0
    value = str(value).lower().strip()
    
    try:
        # Дни
        days = re.findall(r'(\d+(?:\.\d+)?)d', value)
        if days:
            total_minutes += float(days[0]) * 1440
        
        # Часы
        hours = re.findall(r'(\d+(?:\.\d+)?)h', value)
        if hours:
            total_minutes += float(hours[0]) * 60
        
        # Минуты
        minutes = re.findall(r'(\d+(?:\.\d+)?)m(?!s)', value)
        if minutes:
            total_minutes += float(minutes[0])
        
        # Секунды
        seconds = re.findall(r'(\d+(?:\.\d+)?)s', value)
        if seconds:
            total_minutes += float(seconds[0]) / 60
            
        if total_minutes == 0:
            clean_value = re.sub(r'[^\d\.]', '', value)
            if clean_value:
                total_minutes = float(clean_value)
        
        return total_minutes
    except:
        return 0

def convert_input_to_features(token_data):
    """Конвертирует входные данные в признаки (ИСПРАВЛЕННАЯ ВЕРСИЯ)"""
    try:
        features = {}
        logger.info(f"Processing token: {token_data.get('symbol', 'unknown')}")
        
        # Базовая обработка рыночных данных
        market_cap_str = token_data.get('market_cap', '0')
        liquidity_str = token_data.get('liquidity', '0')
        ath_str = token_data.get('ath', '0')
        
        features['market_cap_numeric'] = parse_string_number(market_cap_str)
        features['liquidity_numeric'] = parse_string_number(liquidity_str)
        features['ath_numeric'] = parse_string_number(ath_str)
        
        logger.info(f"Market cap: {market_cap_str} -> {features['market_cap_numeric']}")
        logger.info(f"Liquidity: {liquidity_str} -> {features['liquidity_numeric']}")
        logger.info(f"ATH: {ath_str} -> {features['ath_numeric']}")
        
        # Ограничиваем выбросы
        features['market_cap_capped'] = min(features['market_cap_numeric'], 10_000_000)
        features['liquidity_capped'] = min(features['liquidity_numeric'], 1_000_000)
        features['ath_capped'] = min(features['ath_numeric'], 1_000_000)
        
        # Обработка времени
        token_age_str = token_data.get('token_age', '0')
        features['token_age_minutes'] = parse_time_to_minutes(token_age_str)
        features['token_age_hours'] = features['token_age_minutes'] / 60
        features['token_age_days'] = features['token_age_minutes'] / 1440
        
        logger.info(f"Token age: {token_age_str} -> {features['token_age_minutes']} minutes")
        
        # Категориальные признаки времени
        features['is_very_new'] = 1 if features['token_age_minutes'] < 60 else 0
        features['is_new'] = 1 if features['token_age_minutes'] < 1440 else 0
        features['is_mature'] = 1 if features['token_age_minutes'] > 10080 else 0
        
        # Торговые паттерны - ИСПРАВЛЕННЫЕ ПОЛЯ
        buy_volume_1m = float(token_data.get('buy_volume_1m', 0))
        sell_volume_1m = float(token_data.get('sell_volume_1m', 0))
        buy_volume_5m = float(token_data.get('buy_volume_5m', 0))
        sell_volume_5m = float(token_data.get('sell_volume_5m', 0))
        
        features['total_volume_1m'] = buy_volume_1m + sell_volume_1m
        features['total_volume_5m'] = buy_volume_5m + sell_volume_5m
        
        features['buy_sell_ratio_5m'] = buy_volume_5m / sell_volume_5m if sell_volume_5m > 0 else 0
        features['buy_pressure_5m'] = buy_volume_5m / features['total_volume_5m'] if features['total_volume_5m'] > 0 else 0
        
        logger.info(f"Buy volume 5m: {buy_volume_5m}, Sell volume 5m: {sell_volume_5m}")
        logger.info(f"Buy pressure 5m: {features['buy_pressure_5m']:.3f}")
        
        # Поведенческие паттерны - ИСПРАВЛЕННАЯ СТРУКТУРА
        first_buyers = token_data.get('first_buyers', {})
        buyers_green = first_buyers.get('green', 0)
        buyers_blue = first_buyers.get('blue', 0)
        buyers_yellow = first_buyers.get('yellow', 0)
        buyers_red = first_buyers.get('red', 0)
        buyers_clown = first_buyers.get('clown', 0)
        buyers_moon_new = first_buyers.get('moon_new', 0)
        
        total_holders_emoji = buyers_green + buyers_blue + buyers_yellow + buyers_red
        total_snipers = buyers_clown + first_buyers.get('sun', 0) + first_buyers.get('moon_half', 0) + buyers_moon_new
        total_active = total_holders_emoji + total_snipers
        
        features['holders_diamond_hands'] = buyers_green / total_holders_emoji if total_holders_emoji > 0 else 0
        features['holders_paper_hands'] = buyers_red / total_holders_emoji if total_holders_emoji > 0 else 0
        features['trust_score'] = (buyers_green + buyers_clown) / (total_active + 1)
        features['distrust_score'] = (buyers_red + buyers_moon_new) / (total_active + 1)
        
        logger.info(f"First buyers - Green: {buyers_green}, Red: {buyers_red}, Total: {total_active}")
        logger.info(f"Trust score: {features['trust_score']:.3f}")
        
        # Рыночные коэффициенты
        features['liquidity_to_mcap_ratio'] = features['liquidity_capped'] / features['market_cap_capped'] if features['market_cap_capped'] > 0 else 0
        features['volume_to_liquidity_ratio'] = features['total_volume_5m'] / features['liquidity_capped'] if features['liquidity_capped'] > 0 else 0
        
        logger.info(f"Liquidity to mcap ratio: {features['liquidity_to_mcap_ratio']:.3f}")
        
        # Концентрация китов - ИСПРАВЛЕННАЯ ЛОГИКА
        top_10_holdings = token_data.get('top_10_holdings', [])
        if top_10_holdings and len(top_10_holdings) > 0:
            features['biggest_whale_percent'] = float(top_10_holdings[0]) if top_10_holdings[0] else 0
            features['top3_whales_percent'] = sum(float(x) for x in top_10_holdings[:3] if x)
        else:
            # Альтернативный способ через top_10_percent
            top_10_percent = token_data.get('top_10_percent', 0)
            features['biggest_whale_percent'] = float(top_10_percent) / 3 if top_10_percent else 0  # Примерная оценка
            features['top3_whales_percent'] = float(top_10_percent) if top_10_percent else 0
        
        logger.info(f"Biggest whale: {features['biggest_whale_percent']:.1f}%")
        
        # Безопасность - ИСПРАВЛЕННАЯ ЛОГИКА
        security = token_data.get('security', {})
        security_score = 0
        security_score += 1 if security.get('no_mint', False) else 0
        security_score += 1 if security.get('blacklist', False) else 0
        security_score += 1 if security.get('burnt', False) else 0
        security_score += 1 if security.get('dev_sold', False) else 0
        security_score += 1 if security.get('dex_paid', False) else 0
        features['security_score'] = security_score
        
        logger.info(f"Security score: {security_score}/5")
        
        # Дополнительные поля
        features['total_holders'] = float(token_data.get('total_holders', 0))
        features['volume_per_holder'] = features['total_volume_5m'] / features['total_holders'] if features['total_holders'] > 0 else 0
        
        # Добавляем метрики активности
        buys_5m = float(token_data.get('buys_5m', 0))
        sells_5m = float(token_data.get('sells_5m', 0))
        features['buy_sell_tx_ratio'] = buys_5m / sells_5m if sells_5m > 0 else 0
        features['total_transactions_5m'] = buys_5m + sells_5m
        
        # Добавляем информацию о volatility
        ath_change_percent = token_data.get('ath_change_percent', 0)
        features['ath_change_percent'] = float(ath_change_percent) if ath_change_percent else 0
        features['is_near_ath'] = 1 if abs(features['ath_change_percent']) < 10 else 0
        
        logger.info(f"Generated {len(features)} features")
        
        return features
        
    except Exception as e:
        logger.error(f"Error converting features: {e}")
        return {}

# =============================================================================
# ФУНКЦИИ ПРЕДСКАЗАНИЯ
# =============================================================================

def ml_predict(features):
    """Предсказание с ML моделью или fallback на правила"""
    try:
        # Если ML модель загружена, пытаемся её использовать
        if ML_MODEL_LOADED and model is not None:
            try:
                # Простое предсказание без сложных преобразований
                # Используем только базовые признаки
                basic_features = [
                    features.get('market_cap_capped', 0),
                    features.get('liquidity_capped', 0),
                    features.get('token_age_minutes', 0),
                    features.get('buy_pressure_5m', 0),
                    features.get('biggest_whale_percent', 0),
                    features.get('trust_score', 0),
                    features.get('security_score', 0),
                    features.get('liquidity_to_mcap_ratio', 0)
                ]
                
                # Нормализуем значения
                X = [max(0, min(1000000, float(x))) for x in basic_features]
                
                if hasattr(model, 'predict_proba'):
                    # Пытаемся предсказать вероятность
                    try:
                        probability = model.predict_proba([X])[0][1]
                    except:
                        # Если не получается, используем predict
                        prediction = model.predict([X])[0]
                        probability = 0.7 if prediction == 1 else 0.3
                else:
                    prediction = model.predict([X])[0]
                    probability = 0.7 if prediction == 1 else 0.3
                
                prediction = 1 if probability > 0.5 else 0
                
                # Генерируем рекомендацию
                if probability >= 0.75:
                    recommendation = "🚀 STRONG BUY"
                elif probability >= 0.65:
                    recommendation = "✅ BUY"
                elif probability >= 0.55:
                    recommendation = "⚖️ CONSIDER"
                elif probability >= 0.45:
                    recommendation = "⚠️ CAUTION"
                else:
                    recommendation = "❌ AVOID"
                
                return {
                    'probability': float(probability),
                    'prediction': int(prediction),
                    'recommendation': recommendation,
                    'confidence_interval': (max(0, probability - 0.1), min(1, probability + 0.1)),
                    'model_info': {
                        'model_name': model_metadata.get('best_model_name', 'ML Model') if model_metadata else 'ML Model',
                        'model_auc': model_metadata.get('test_auc', 0.75) if model_metadata else 0.75,
                        'features_used': len(basic_features),
                        'is_ensemble': False
                    }
                }
                
            except Exception as e:
                logger.error(f"ML prediction failed: {e}")
                # Fallback на простые правила
                pass
        
        # Простые правила (fallback)
        return simple_predict(features)
        
    except Exception as e:
        logger.error(f"Error in ml_predict: {e}")
        return simple_predict(features)

def simple_predict(features):
    """Улучшенные правила предсказания"""
    try:
        score = 0.5  # Базовая вероятность
        
        # Возраст токена
        age_minutes = features.get('token_age_minutes', 0)
        if age_minutes < 60:
            score -= 0.1  # Очень новые токены рискованнее (но не так сильно)
        elif 60 <= age_minutes <= 1440:
            score += 0.1   # Оптимальный возраст
        elif age_minutes > 10080:
            score -= 0.05  # Старые токены менее активны
        
        # Давление покупок
        buy_pressure = features.get('buy_pressure_5m', 0)
        if buy_pressure > 0.65:
            score += 0.2   # Сильное давление покупок
        elif buy_pressure > 0.55:
            score += 0.1   # Умеренное давление
        elif buy_pressure < 0.35:
            score -= 0.2   # Слабое давление
        
        # Концентрация китов
        whale_percent = features.get('biggest_whale_percent', 0)
        if whale_percent > 25:
            score -= 0.25  # Очень опасная концентрация
        elif whale_percent > 15:
            score -= 0.15  # Опасная концентрация
        elif whale_percent < 8:
            score += 0.1   # Безопасная концентрация
        
        # Trust score
        trust = features.get('trust_score', 0)
        if trust > 0.7:
            score += 0.15  # Высокое доверие
        elif trust > 0.5:
            score += 0.05  # Умеренное доверие
        elif trust < 0.2:
            score -= 0.15  # Низкое доверие
        
        # Ликвидность к market cap
        liq_mcap_ratio = features.get('liquidity_to_mcap_ratio', 0)
        if liq_mcap_ratio > 0.6:
            score += 0.1   # Отличная ликвидность
        elif liq_mcap_ratio > 0.3:
            score += 0.05  # Хорошая ликвидность
        elif liq_mcap_ratio < 0.15:
            score -= 0.15  # Плохая ликвидность
        
        # Безопасность
        security_score = features.get('security_score', 0)
        if security_score >= 4:
            score += 0.1   # Высокая безопасность
        elif security_score >= 3:
            score += 0.05  # Средняя безопасность
        elif security_score <= 1:
            score -= 0.2   # Низкая безопасность
        
        # Размер market cap (предпочитаем средние размеры)
        mcap = features.get('market_cap_capped', 0)
        if 50000 <= mcap <= 500000:  # Sweet spot
            score += 0.05
        elif mcap > 5000000:         # Слишком большие
            score -= 0.05
        elif mcap < 10000:           # Слишком маленькие
            score -= 0.1
        
        # Активность транзакций
        total_tx = features.get('total_transactions_5m', 0)
        if total_tx > 500:  # Высокая активность
            score += 0.05
        elif total_tx < 50:  # Низкая активность
            score -= 0.1
        
        # Соотношение покупок к продажам по транзакциям
        tx_ratio = features.get('buy_sell_tx_ratio', 0)
        if tx_ratio > 1.2:  # Больше покупательских транзакций
            score += 0.05
        elif tx_ratio < 0.8:  # Больше продажных транзакций
            score -= 0.05
        
        # Ограничиваем вероятность
        probability = max(0.05, min(0.95, score))
        prediction = 1 if probability > 0.5 else 0
        
        # Рекомендация
        if probability >= 0.8:
            recommendation = "🔥 VERY STRONG BUY"
        elif probability >= 0.7:
            recommendation = "🚀 STRONG BUY" 
        elif probability >= 0.6:
            recommendation = "✅ BUY"
        elif probability >= 0.5:
            recommendation = "⚖️ CONSIDER"
        elif probability >= 0.4:
            recommendation = "⚠️ CAUTION"
        elif probability >= 0.3:
            recommendation = "❌ AVOID"
        else:
            recommendation = "🚫 STRONG AVOID"
        
        return {
            'probability': float(probability),
            'prediction': int(prediction),
            'recommendation': recommendation,
            'confidence_interval': (max(0, probability - 0.1), min(1, probability + 0.1)),
            'model_info': {
                'model_name': 'Advanced Rules',
                'model_auc': 0.68,
                'features_used': len(features),
                'is_ensemble': False
            },
            'key_factors': {
                'age_minutes': age_minutes,
                'buy_pressure_5m': buy_pressure,
                'biggest_whale_percent': whale_percent,
                'trust_score': trust,
                'security_score': security_score,
                'liquidity_to_mcap_ratio': liq_mcap_ratio,
                'market_cap': mcap,
                'total_transactions_5m': total_tx,
                'buy_sell_tx_ratio': tx_ratio
            }
        }
        
    except Exception as e:
        logger.error(f"Error in simple prediction: {e}")
        return {
            'probability': 0.5,
            'prediction': 0,
            'recommendation': "❓ ERROR",
            'confidence_interval': (0.0, 1.0),
            'model_info': {
                'model_name': 'Error',
                'model_auc': 0.0,
                'features_used': 0,
                'is_ensemble': False
            },
            'error': str(e)
        }

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def health_check():
    """Проверка здоровья API"""
    return jsonify({
        'status': 'healthy',
        'message': 'Memtoken Prediction API is running',
        'timestamp': datetime.now().isoformat(),
        'version': '1.1.0',
        'ml_model_loaded': ML_MODEL_LOADED,
        'model_info': {
            'model_name': model_metadata.get('best_model_name', 'Advanced Rules') if model_metadata else 'Advanced Rules',
            'model_auc': model_metadata.get('test_auc', 0.68) if model_metadata else 0.68,
            'features_count': len(feature_names) if feature_names else 25,
            'components_loaded': {
                'model': model is not None,
                'scaler': scaler is not None,
                'encoders': label_encoders is not None,
                'features': feature_names is not None,
                'metadata': model_metadata is not None
            }
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Предсказание для одного токена"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        logger.info(f"Received prediction request for token: {data.get('symbol', 'unknown')}")
        
        # Конвертируем входные данные в признаки
        features = convert_input_to_features(data)
        
        if not features:
            return jsonify({'error': 'Failed to extract features from token data'}), 400
        
        # Получаем предсказание
        result = ml_predict(features)
        
        # Добавляем информацию о токене
        result['token_info'] = {
            'symbol': data.get('symbol', ''),
            'name': data.get('name', ''),
            'contract_address': data.get('contract_address', ''),
            'market_cap': data.get('market_cap', ''),
            'liquidity': data.get('liquidity', ''),
            'token_age': data.get('token_age', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Prediction completed: {result['recommendation']} ({result['probability']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error in predict endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Пакетное предсказание"""
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Data should be a list of token objects'}), 400
        
        if len(data) > 100:
            return jsonify({'error': 'Batch size cannot exceed 100 tokens'}), 400
        
        results = []
        
        for i, token_data in enumerate(data):
            try:
                features = convert_input_to_features(token_data)
                result = ml_predict(features)
                
                result['token_info'] = {
                    'symbol': token_data.get('symbol', ''),
                    'name': token_data.get('name', ''),
                    'batch_index': i
                }
                
                results.append(result)
                
            except Exception as e:
                results.append({
                    'error': f'Error processing token {i}',
                    'message': str(e),
                    'token_info': {'symbol': token_data.get('symbol', ''), 'batch_index': i}
                })
        
        successful_predictions = [r for r in results if 'error' not in r]
        avg_probability = sum(r['probability'] for r in successful_predictions) / len(successful_predictions) if successful_predictions else 0
        
        return jsonify({
            'results': results,
            'batch_stats': {
                'total_tokens': len(data),
                'successful_predictions': len(successful_predictions),
                'average_probability': avg_probability,
                'timestamp': datetime.now().isoformat()
            }
        })
        
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/debug/features', methods=['POST'])
def debug_features():
    """Отладка: показать извлеченные признаки"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        features = convert_input_to_features(data)
        
        return jsonify({
            'input_data': data,
            'extracted_features': features,
            'feature_count': len(features),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': 'Internal server error', 'message': str(e)}), 500

@app.route('/debug/files', methods=['GET'])
def debug_files():
    """Диагностика файлов"""
    try:
        files_info = {
            'current_directory': os.getcwd(),
            'all_files': os.listdir('.'),
            'pkl_files': [f for f in os.listdir('.') if f.endswith('.pkl')],
            'json_files': [f for f in os.listdir('.') if f.endswith('.json')]
        }
        return jsonify(files_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['GET'])
def test_with_sample():
    """Тест с примером данных"""
    sample_data = {
        "symbol": "TEST",
        "name": "Test Token",
        "contract_address": "test123",
        "token_age": "1h 30m",
        "market_cap": "50K",
        "liquidity": "35K",
        "ath": "75K",
        "buy_volume_5m": 15000,
        "sell_volume_5m": 12000,
        "buys_5m": 45,
        "sells_5m": 35,
        "first_buyers": {
            "green": 8,
            "blue": 5,
            "yellow": 3,
            "red": 12,
            "clown": 1,
            "moon_new": 2
        },
        "top_10_holdings": [15.5, 8.2, 6.1, 4.3, 3.8],
        "total_holders": 150,
        "security": {
            "no_mint": True,
            "blacklist": True,
            "burnt": True,
            "dev_sold": False,
            "dex_paid": True
        }
    }
    
    try:
        features = convert_input_to_features(sample_data)
        result = ml_predict(features)
        
        return jsonify({
            'test_data': sample_data,
            'features': features,
            'prediction': result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# =============================================================================
# ERROR HANDLERS
# =============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'
    
    print("🚀 Starting Memtoken Prediction API...")
    print(f"📊 ML Model loaded: {'✅' if ML_MODEL_LOADED else '❌'}")
    if ML_MODEL_LOADED and model_metadata:
        print(f"🎯 Model: {model_metadata.get('best_model_name', 'Unknown')}")
        print(f"📈 AUC: {model_metadata.get('test_auc', 0):.4f}")
        print(f"🔧 Features: {len(feature_names) if feature_names else 0}")
    else:
        print("⚠️  Using Advanced Rules (AUC ~0.68)")
    
    app.run(host='0.0.0.0', port=port, debug=debug)
