# app.py - Простой API для Railway + n8n
import os
import pickle
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Глобальная переменная для модели
model_artifacts = None

def load_model():
    """Загружает модель при старте приложения"""
    global model_artifacts
    try:
        model_file = 'solana_token_xgboost_model.pkl'
        if os.path.exists(model_file):
            with open(model_file, 'rb') as f:
                model_artifacts = pickle.load(f)
            logger.info(f"✅ Модель загружена! AUC: {model_artifacts['performance_metrics']['test_auc']:.4f}")
            return True
        else:
            logger.error(f"❌ Файл модели {model_file} не найден")
            return False
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        return False

def predict_token_success(token_data):
    """Функция предсказания успешности токена"""
    
    if model_artifacts is None:
        raise ValueError("Модель не загружена")
    
    try:
        # Создаем DataFrame из входных данных
        df_new = pd.DataFrame([token_data])
        
        # Добавляем недостающие столбцы с нулевыми значениями
        for col in model_artifacts['feature_names']:
            if col not in df_new.columns:
                df_new[col] = 0
        
        # Упорядочиваем столбцы
        df_new = df_new[model_artifacts['feature_names']]
        
        # Применяем импутер
        df_imputed = pd.DataFrame(
            model_artifacts['imputer'].transform(df_new), 
            columns=model_artifacts['feature_names']
        )
        
        # Получаем предсказания
        prediction = model_artifacts['model'].predict(df_imputed)[0]
        probability = model_artifacts['model'].predict_proba(df_imputed)[0, 1]
        
        # Определяем уверенность
        confidence_score = abs(probability - 0.5) * 2
        if confidence_score > 0.8:
            confidence_level = "very_high"
        elif confidence_score > 0.6:
            confidence_level = "high"
        elif confidence_score > 0.4:
            confidence_level = "medium"
        else:
            confidence_level = "low"
        
        # Формируем результат
        result = {
            'prediction': 'success' if prediction == 1 else 'fail',
            'binary_prediction': int(prediction),
            'probability': round(probability, 4),
            'probability_percent': round(probability * 100, 1),
            'confidence_score': round(confidence_score, 4),
            'confidence_level': confidence_level,
            'expected_pnl': 'PNL >= 2x' if prediction == 1 else 'PNL < 2x'
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}")
        raise

def analyze_token_signals(token_data):
    """Анализирует сигналы токена"""
    
    # Позитивные сигналы (холдят и докупают)
    positive = (token_data.get('buyers_green', 0) + 
                token_data.get('buyers_blue', 0) + 
                token_data.get('buyers_clown', 0) + 
                token_data.get('buyers_sun', 0))
    
    # Негативные сигналы (продали полностью)
    negative = (token_data.get('buyers_red', 0) + 
                token_data.get('buyers_moon_new', 0))
    
    # Соотношение
    ratio = positive / (negative + 1)
    
    if ratio > 3:
        status = "very_good"
    elif ratio > 1.5:
        status = "good"
    elif ratio > 0.8:
        status = "medium"
    else:
        status = "bad"
    
    return {
        'positive_signals': positive,
        'negative_signals': negative,
        'signal_ratio': round(ratio, 2),
        'signal_status': status
    }

def check_token_safety(token_data):
    """Проверяет безопасность токена"""
    
    safety_score = 0
    issues = []
    
    # Проверяем концентрацию
    top10 = token_data.get('top_10_percent', 0)
    if top10 <= 70:
        safety_score += 1
    else:
        issues.append(f"high_concentration_{top10}")
    
    # Проверяем разработчика
    dev_percent = token_data.get('dev_current_balance_percent', 0)
    if dev_percent <= 20:
        safety_score += 1
    else:
        issues.append(f"dev_holds_much_{dev_percent}")
    
    # Проверяем ликвидность
    liquidity = token_data.get('liquidity', 0)
    if liquidity >= 100000:
        safety_score += 1
    else:
        issues.append(f"low_liquidity_{liquidity}")
    
    # Проверяем безопасность
    if token_data.get('security_no_mint', 0):
        safety_score += 1
    else:
        issues.append("mint_not_disabled")
        
    if token_data.get('security_burnt', 0):
        safety_score += 1
    else:
        issues.append("tokens_not_burnt")
    
    return {
        'safety_score': safety_score,
        'max_safety_score': 5,
        'safety_percentage': round((safety_score / 5) * 100),
        'safety_issues': issues,
        'is_safe': safety_score >= 3
    }

def get_recommendation(prediction_result, signals, safety):
    """Генерирует финальную рекомендацию"""
    
    probability = prediction_result['probability']
    safety_score = safety['safety_score']
    
    if probability >= 0.7 and safety_score >= 4:
        return {
            'recommendation': 'BUY',
            'risk_level': 'low',
            'reason': 'excellent_token'
        }
    elif probability >= 0.6 and safety_score >= 3:
        return {
            'recommendation': 'CONSIDER',
            'risk_level': 'medium',
            'reason': 'good_potential'
        }
    elif probability >= 0.5 and safety_score >= 2:
        return {
            'recommendation': 'CAUTION',
            'risk_level': 'medium_high',
            'reason': 'moderate_risk'
        }
    else:
        return {
            'recommendation': 'AVOID',
            'risk_level': 'high',
            'reason': 'high_risk'
        }

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.route('/', methods=['GET'])
def home():
    """Главная страница API"""
    return jsonify({
        'service': 'Solana Token Predictor API',
        'status': 'online',
        'model_loaded': model_artifacts is not None,
        'endpoints': {
            'predict': '/predict [POST]',
            'health': '/health [GET]',
            'model_info': '/model-info [GET]'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Основной endpoint для предсказания токенов"""
    
    if model_artifacts is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Получаем JSON данные
        if not request.is_json:
            return jsonify({
                'success': False,
                'error': 'Content-Type must be application/json'
            }), 400
        
        token_data = request.get_json()
        
        if not token_data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Делаем предсказание
        prediction_result = predict_token_success(token_data)
        
        # Анализируем сигналы
        signals = analyze_token_signals(token_data)
        
        # Проверяем безопасность
        safety = check_token_safety(token_data)
        
        # Получаем рекомендацию
        recommendation = get_recommendation(prediction_result, signals, safety)
        
        # Формируем ответ
        response = {
            'success': True,
            'token_symbol': token_data.get('symbol', 'UNKNOWN'),
            'prediction': prediction_result,
            'signals': signals,
            'safety': safety,
            'recommendation': recommendation,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"✅ Предсказание: {token_data.get('symbol', 'UNKNOWN')} -> {prediction_result['prediction']} ({prediction_result['probability_percent']}%)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check для Railway"""
    
    model_info = {}
    if model_artifacts:
        model_info = {
            'auc': model_artifacts['performance_metrics']['test_auc'],
            'f1': model_artifacts['performance_metrics']['test_f1'],
            'features_count': len(model_artifacts['feature_names'])
        }
    
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_artifacts is not None,
        'model_info': model_info,
        'timestamp': pd.Timestamp.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Информация о модели"""
    
    if model_artifacts is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    # Топ-10 важных признаков
    top_features = model_artifacts['feature_importance'].head(10).to_dict('records')
    
    return jsonify({
        'success': True,
        'model_type': model_artifacts.get('model_type', 'Unknown'),
        'training_approach': model_artifacts.get('training_approach', 'Unknown'),
        'performance_metrics': model_artifacts['performance_metrics'],
        'features_count': len(model_artifacts['feature_names']),
        'top_features': top_features,
        'all_features': model_artifacts['feature_names']
    })

@app.route('/example', methods=['GET'])
def example():
    """Пример входных данных для n8n"""
    
    example_data = {
        "symbol": "BONK",
        "market_cap": 2000000,
        "liquidity": 800000,
        "volume_1m": 100000,
        "buy_volume_1m": 65000,
        "sell_volume_1m": 35000,
        "buyers_green": 150,
        "buyers_blue": 30,
        "buyers_yellow": 20,
        "buyers_red": 15,
        "buyers_clown": 8,
        "buyers_sun": 3,
        "buyers_moon_half": 1,
        "buyers_moon_new": 2,
        "total_holders": 223,
        "top_10_percent": 35.0,
        "dev_current_balance_percent": 5.0,
        "security_no_mint": 1,
        "security_burnt": 1,
        "security_dev_sold": 0,
        "token_age_minutes": 180
    }
    
    return jsonify({
        'example_request': {
            'url': request.base_url.replace('/example', '/predict'),
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': example_data
        },
        'required_fields': [
            'market_cap', 'liquidity', 'buyers_green', 'buyers_red'
        ],
        'optional_fields': [
            'symbol', 'volume_1m', 'buyers_blue', 'buyers_yellow',
            'buyers_clown', 'buyers_sun', 'buyers_moon_half', 'buyers_moon_new',
            'total_holders', 'top_10_percent', 'dev_current_balance_percent',
            'security_no_mint', 'security_burnt', 'security_dev_sold',
            'token_age_minutes'
        ]
    })

# Обработка ошибок
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/predict', '/health', '/model-info', '/example']
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Запуск приложения
if __name__ == '__main__':
    # Загружаем модель при старте
    load_model()
    
    # Получаем порт из переменной окружения (Railway)
    port = int(os.environ.get('PORT', 5000))
    
    # Запускаем приложение
    app.run(host='0.0.0.0', port=port, debug=False)
