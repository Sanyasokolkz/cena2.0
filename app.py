# app.py - API для Railway + n8n со структурированными JSON данными
import os
import pickle
import pandas as pd
import numpy as np
import re
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

def parse_value_with_suffix(value_str):
    """Парсит строковые значения с суффиксами K, M, B"""
    if not value_str or value_str is None:
        return 0
    
    if isinstance(value_str, (int, float)):
        return float(value_str)
    
    value_str = str(value_str).replace(',', '').replace('
import os
import pickle
import pandas as pd
import numpy as np
import re
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

def parse_token_data(cena_full_text):
    """Парсит текстовые данные токена в структурированный формат"""
    
    try:
        # Извлекаем символ токена
        symbol_match = re.search(r'\$([A-Z0-9]+)', cena_full_text)
        symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"
        
        # Извлекаем возраст токена (в минутах)
        age_match = re.search(r'Token age:\s*(\d+)m', cena_full_text)
        token_age_minutes = int(age_match.group(1)) if age_match else 0
        
        # Извлекаем рыночные данные
        mc_match = re.search(r'MC: \$([0-9,\.]+)K', cena_full_text)
        market_cap = float(mc_match.group(1).replace(',', '')) * 1000 if mc_match else 0
        
        liq_match = re.search(r'Liq: \$([0-9,\.]+)K', cena_full_text)
        liquidity = float(liq_match.group(1).replace(',', '')) * 1000 if liq_match else 0
        
        # SOL pooled
        sol_match = re.search(r'SOL pooled: ([0-9,\.]+)', cena_full_text)
        sol_pooled = float(sol_match.group(1).replace(',', '')) if sol_match else 0
        
        # ATH
        ath_match = re.search(r'ATH: \$([0-9,\.]+)K', cena_full_text)
        ath = float(ath_match.group(1).replace(',', '')) * 1000 if ath_match else 0
        
        # Объемы 1 минута
        vol_1m_match = re.search(r'Volume: \$([0-9,\.]+)', cena_full_text)
        volume_1m = float(vol_1m_match.group(1).replace(',', '')) if vol_1m_match else 0
        
        buy_vol_1m_match = re.search(r'Buy volume \(\$\): \$([0-9,\.]+)', cena_full_text)
        buy_volume_1m = float(buy_vol_1m_match.group(1).replace(',', '')) if buy_vol_1m_match else 0
        
        sell_vol_1m_match = re.search(r'Sell volume \(\$\): \$([0-9,\.]+)', cena_full_text)
        sell_volume_1m = float(sell_vol_1m_match.group(1).replace(',', '')) if sell_vol_1m_match else 0
        
        # Количество покупок/продаж
        buys_1m_match = re.search(r'Buys: (\d+)', cena_full_text)
        buys_1m = int(buys_1m_match.group(1)) if buys_1m_match else 0
        
        sells_1m_match = re.search(r'Sells: (\d+)', cena_full_text)
        sells_1m = int(sells_1m_match.group(1)) if sells_1m_match else 0
        
        # Объемы 5 минут (ищем вторые значения)
        vol_5m_matches = re.findall(r'Volume: \$([0-9,\.]+)', cena_full_text)
        volume_5m = float(vol_5m_matches[1].replace(',', '')) if len(vol_5m_matches) > 1 else 0
        
        buy_vol_5m_matches = re.findall(r'Buy volume \(\$\): \$([0-9,\.]+)', cena_full_text)
        buy_volume_5m = float(buy_vol_5m_matches[1].replace(',', '')) if len(buy_vol_5m_matches) > 1 else 0
        
        sell_vol_5m_matches = re.findall(r'Sell volume \(\$\): \$([0-9,\.]+)', cena_full_text)
        sell_volume_5m = float(sell_vol_5m_matches[1].replace(',', '')) if len(sell_vol_5m_matches) > 1 else 0
        
        buys_5m_matches = re.findall(r'Buys: (\d+)', cena_full_text)
        buys_5m = int(buys_5m_matches[1]) if len(buys_5m_matches) > 1 else 0
        
        sells_5m_matches = re.findall(r'Sells: (\d+)', cena_full_text)
        sells_5m = int(sells_5m_matches[1]) if len(sells_5m_matches) > 1 else 0
        
        # Покупатели по категориям
        green_match = re.search(r'🟢: (\d+)', cena_full_text)
        buyers_green = int(green_match.group(1)) if green_match else 0
        
        blue_match = re.search(r'🔵: (\d+)', cena_full_text)
        buyers_blue = int(blue_match.group(1)) if blue_match else 0
        
        yellow_match = re.search(r'🟡: (\d+)', cena_full_text)
        buyers_yellow = int(yellow_match.group(1)) if yellow_match else 0
        
        red_match = re.search(r'⭕️: (\d+)', cena_full_text)
        buyers_red = int(red_match.group(1)) if red_match else 0
        
        # Снайперы
        clown_match = re.search(r'🤡: (\d+)', cena_full_text)
        buyers_clown = int(clown_match.group(1)) if clown_match else 0
        
        sun_match = re.search(r'🌞: (\d+)', cena_full_text)
        buyers_sun = int(sun_match.group(1)) if sun_match else 0
        
        moon_half_match = re.search(r'🌗: (\d+)', cena_full_text)
        buyers_moon_half = int(moon_half_match.group(1)) if moon_half_match else 0
        
        moon_new_match = re.search(r'🌚: (\d+)', cena_full_text)
        buyers_moon_new = int(moon_new_match.group(1)) if moon_new_match else 0
        
        # Current/Initial ratio
        ratio_match = re.search(r'Current/Initial: ([0-9\.]+)% / ([0-9\.]+)%', cena_full_text)
        current_ratio = float(ratio_match.group(1)) if ratio_match else 0
        initial_ratio = float(ratio_match.group(2)) if ratio_match else 0
        
        # Держатели
        total_holders_match = re.search(r'Total: (\d+)', cena_full_text)
        total_holders = int(total_holders_match.group(1)) if total_holders_match else 0
        
        freshies_1d_match = re.search(r'Freshies: ([0-9\.]+)% 1D', cena_full_text)
        freshies_1d_percent = float(freshies_1d_match.group(1)) if freshies_1d_match else 0
        
        freshies_7d_match = re.search(r'([0-9\.]+)% 7D', cena_full_text)
        freshies_7d_percent = float(freshies_7d_match.group(1)) if freshies_7d_match else 0
        
        top_10_match = re.search(r'Top 10: (\d+)%', cena_full_text)
        top_10_percent = float(top_10_match.group(1)) if top_10_match else 0
        
        # Top 10 holdings - берем первое значение
        holdings_match = re.search(r'Top 10 Holding.*?\n([0-9\.]+)', cena_full_text, re.DOTALL)
        top_10_holdings = float(holdings_match.group(1)) if holdings_match else 0
        
        # Dev данные
        dev_balance_match = re.search(r'Dev current balance: (\d+)%', cena_full_text)
        dev_current_balance_percent = float(dev_balance_match.group(1)) if dev_balance_match else 0
        
        dev_sol_match = re.search(r'Dev SOL balance: ([0-9\.]+) SOL', cena_full_text)
        dev_sol_balance = float(dev_sol_match.group(1)) if dev_sol_match else 0
        
        # Security flags (🟢 = 1, 🔴 = 0)
        security_no_mint = 1 if '├ NoMint: 🟢' in cena_full_text else 0
        security_blacklist = 1 if '├ Blacklist: 🟢' in cena_full_text else 0
        security_burnt = 1 if '├ Burnt: 🟢' in cena_full_text else 0
        security_dev_sold = 1 if '├ Dev Sold: 🟢' in cena_full_text else 0
        security_dex_paid = 1 if '└ Dex Paid: 🟢' in cena_full_text else 0
        
        # Формируем структурированные данные
        parsed_data = {
            'symbol': symbol,
            'token_age_minutes': token_age_minutes,
            'market_cap': market_cap,
            'liquidity': liquidity,
            'sol_pooled': sol_pooled,
            'ath': ath,
            'volume_1m': volume_1m,
            'buy_volume_1m': buy_volume_1m,
            'sell_volume_1m': sell_volume_1m,
            'buys_1m': buys_1m,
            'sells_1m': sells_1m,
            'volume_5m': volume_5m,
            'buy_volume_5m': buy_volume_5m,
            'sell_volume_5m': sell_volume_5m,
            'buys_5m': buys_5m,
            'sells_5m': sells_5m,
            'buyers_green': buyers_green,
            'buyers_blue': buyers_blue,
            'buyers_yellow': buyers_yellow,
            'buyers_red': buyers_red,
            'buyers_clown': buyers_clown,
            'buyers_sun': buyers_sun,
            'buyers_moon_half': buyers_moon_half,
            'buyers_moon_new': buyers_moon_new,
            'current_ratio': current_ratio,
            'initial_ratio': initial_ratio,
            'total_holders': total_holders,
            'freshies_1d_percent': freshies_1d_percent,
            'freshies_7d_percent': freshies_7d_percent,
            'top_10_percent': top_10_percent,
            'top_10_holdings': top_10_holdings,
            'dev_current_balance_percent': dev_current_balance_percent,
            'dev_sol_balance': dev_sol_balance,
            'security_no_mint': security_no_mint,
            'security_blacklist': security_blacklist,
            'security_burnt': security_burnt,
            'security_dev_sold': security_dev_sold,
            'security_dex_paid': security_dex_paid
        }
        
        logger.info(f"✅ Парсинг успешен для {symbol}: MC=${market_cap:,.0f}, Liq=${liquidity:,.0f}")
        return parsed_data
        
    except Exception as e:
        logger.error(f"❌ Ошибка парсинга: {e}")
        raise ValueError(f"Ошибка парсинга данных токена: {e}")

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
        
        # Получаем порт из переменной окружения (Railway)
    port = int(os.environ.get('PORT', 5000))
    
    # Запускаем приложение
    app.run(host='0.0.0.0', port=port, debug=False)аем предсказания
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
            'predict_text': '/predict-text [POST]',
            'health': '/health [GET]',
            'model_info': '/model-info [GET]'
        }
    })

@app.route('/predict-text', methods=['POST'])
def predict_text():
    """Endpoint для предсказания из текстовых данных (основной для n8n)"""
    
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
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Ожидаем поле cena_full с текстовыми данными
        cena_full_text = data.get('cena_full')
        if not cena_full_text:
            return jsonify({
                'success': False,
                'error': 'Field "cena_full" is required'
            }), 400
        
        # Парсим текстовые данные
        parsed_token_data = parse_token_data(cena_full_text)
        
        # Делаем предсказание
        prediction_result = predict_token_success(parsed_token_data)
        
        # Анализируем сигналы
        signals = analyze_token_signals(parsed_token_data)
        
        # Проверяем безопасность
        safety = check_token_safety(parsed_token_data)
        
        # Получаем рекомендацию
        recommendation = get_recommendation(prediction_result, signals, safety)
        
        # Формируем ответ
        response = {
            'success': True,
            'token_symbol': parsed_token_data.get('symbol', 'UNKNOWN'),
            'parsed_data': parsed_token_data,
            'prediction': prediction_result,
            'signals': signals,
            'safety': safety,
            'recommendation': recommendation,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"✅ Предсказание: {parsed_token_data.get('symbol', 'UNKNOWN')} -> {prediction_result['prediction']} ({prediction_result['probability_percent']}%)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint для предсказания из структурированных данных"""
    
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

@app.route('/example-text', methods=['GET'])
def example_text():
    """Пример входных данных в текстовом формате"""
    
    example_data = {
        "cena_full": "🎲 $CRUMB | Crumbcat\nHz7MeU72BNF9rCWyUFAwKTyCcjr6qsJm1jwYehnqjups\n⏳ Token age:  2m  | 👁 59\n├ MC: $129.1K\n├ Liq: $47.3K / SOL pooled: --\n└ ATH: $152.7K (-21% / 32s)\n1 min:\n├ Volume: $148,947.33\n├ Buy volume ($): $81,396.06\n├ Sell volume ($): $67,551.28\n├ Buys: 567\n└ Sells: 453\n5 min:\n├ Volume: $172,794.94\n├ Buy volume ($): $98,494.49\n├ Sell volume ($): $74,300.45\n├ Buys: 670\n└ Sells: 517\n🎯 First 70 buyers:\n⭕️⭕️⭕️⭕️🟡⭕️🔵⭕️⭕️🟡\n🟡🟡⭕️🟡⭕️🟡⭕️⭕️⭕️⭕️\n⭕️🟡🟡⭕️🟢🟡🟡⭕️⭕️⭕️\n🟡🟡🟢🟢⭕️🟡🟡🟢🟡🟡\n🟡⭕️🟡🟡🟢🟢🟢⭕️⭕️🟡\n🟢🟡⭕️🟡🟡🟢🟡🔵🟡⭕️\n⭕️🟢🟡🟢⭕️⭕️🟢🟢🟢🟡\n├ 🟢: 14 | 🔵: 2 | 🟡: 27 | ⭕️: 27\n├ 🤡: 0 | 🌞: 0 | 🌗: 0 | 🌚: 0\n├ Current/Initial: 23.8% / 72.58%\n👥 Holders:\n├ Total: 383\n├ Freshies: 5.5% 1D | 18% 7D\n├ Top 10: 21%\n💰 Top 10 Holding (%)\n29.2 | 3.38 | 3.32 | 2.96 | 2.95 | 2.56 | 2.49 | 2.26 | 1.95 | 1.89\n😎 Dev\n├ Dev current balance: 0%\n└ Dev SOL balance: 0.225 SOL\n🔒 Security:\n├ NoMint: 🟢\n├ Blacklist: 🟢\n├ Burnt: 🟢\n├ Dev Sold: 🟢\n└ Dex Paid: 🔴"
    }
    
    return jsonify({
        'example_request': {
            'url': request.base_url.replace('/example-text', '/predict-text'),
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': example_data
        },
        'required_fields': [
            'cena_full (string with full token data)'
        ],
        'note': 'Send the complete text data in cena_full field'
    })

# Обработка ошибок
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/predict-text', '/predict', '/health', '/model-info', '/example-text']
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
    
    # Получ# app.py - Простой API для Railway + n8n
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
    app.run(host='0.0.0.0', port=port, debug=False), '').strip()
    
    if value_str.endswith('K'):
        return float(value_str[:-1]) * 1000
    elif value_str.endswith('M'):
        return float(value_str[:-1]) * 1000000
    elif value_str.endswith('B'):
        return float(value_str[:-1]) * 1000000000
    else:
        try:
            return float(value_str)
        except:
            return 0

def parse_token_age_minutes(token_age_str):
    """Парсит возраст токена в минуты"""
    if not token_age_str:
        return 0
    
    # Извлекаем число и единицу измерения
    match = re.match(r'(\d+)([mhd]?)', str(token_age_str))
    if not match:
        return 0
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == 'm' or unit == '':  # минуты
        return value
    elif unit == 'h':  # часы
        return value * 60
    elif unit == 'd':  # дни
        return value * 60 * 24
    else:
        return value

def process_token_data(token_json):
    """Обрабатывает структурированные JSON данные токена"""
    
    try:
        # Базовая информация
        symbol = token_json.get('symbol', 'UNKNOWN')
        token_age_minutes = parse_token_age_minutes(token_json.get('token_age'))
        
        # Рыночные данные
        market_cap = parse_value_with_suffix(token_json.get('market_cap'))
        liquidity = parse_value_with_suffix(token_json.get('liquidity'))
        ath = parse_value_with_suffix(token_json.get('ath'))
        sol_pooled = token_json.get('sol_pooled') or 0
        
        # Объемы торгов
        volume_1m = token_json.get('volume_1m', 0)
        buy_volume_1m = token_json.get('buy_volume_1m', 0)
        sell_volume_1m = token_json.get('sell_volume_1m', 0)
        buys_1m = token_json.get('buys_1m', 0)
        sells_1m = token_json.get('sells_1m', 0)
        
        volume_5m = token_json.get('volume_5m', 0)
        buy_volume_5m = token_json.get('buy_volume_5m', 0)
        sell_volume_5m = token_json.get('sell_volume_5m', 0)
        buys_5m = token_json.get('buys_5m', 0)
        sells_5m = token_json.get('sells_5m', 0)
        
        # Покупатели (из first_buyers)
        first_buyers = token_json.get('first_buyers', {})
        buyers_green = first_buyers.get('green', 0)
        buyers_blue = first_buyers.get('blue', 0)
        buyers_yellow = first_buyers.get('yellow', 0)
        buyers_red = first_buyers.get('red', 0)
        buyers_clown = first_buyers.get('clown', 0)
        buyers_sun = first_buyers.get('sun', 0)
        buyers_moon_half = first_buyers.get('moon_half', 0)
        buyers_moon_new = first_buyers.get('moon_new', 0)
        
        # Current/Initial ratio
        ratio_data = token_json.get('current_initial_ratio', {})
        current_ratio = ratio_data.get('current', 0)
        initial_ratio = ratio_data.get('initial', 0)
        
        # Держатели
        total_holders = token_json.get('total_holders', 0)
        freshies_1d_percent = token_json.get('freshies_1d_percent', 0)
        freshies_7d_percent = token_json.get('freshies_7d_percent', 0)
        top_10_percent = token_json.get('top_10_percent', 0)
        
        # Top 10 holdings - берем первое значение
        top_10_holdings_list = token_json.get('top_10_holdings', [])
        top_10_holdings = top_10_holdings_list[0] if top_10_holdings_list else 0
        
        # Dev данные
        dev_current_balance_percent = token_json.get('dev_current_balance_percent', 0)
        dev_sol_balance = token_json.get('dev_sol_balance', 0)
        
        # Security данные
        security = token_json.get('security', {})
        security_no_mint = 1 if security.get('no_mint') else 0
        security_blacklist = 1 if security.get('blacklist') else 0
        security_burnt = 1 if security.get('burnt') else 0
        security_dev_sold = 1 if security.get('dev_sold') else 0
        security_dex_paid = 1 if security.get('dex_paid') else 0
        
        # Формируем структурированные данные для модели
        processed_data = {
            'symbol': symbol,
            'token_age_minutes': token_age_minutes,
            'market_cap': market_cap,
            'liquidity': liquidity,
            'sol_pooled': sol_pooled,
            'ath': ath,
            'volume_1m': volume_1m,
            'buy_volume_1m': buy_volume_1m,
            'sell_volume_1m': sell_volume_1m,
            'buys_1m': buys_1m,
            'sells_1m': sells_1m,
            'volume_5m': volume_5m,
            'buy_volume_5m': buy_volume_5m,
            'sell_volume_5m': sell_volume_5m,
            'buys_5m': buys_5m,
            'sells_5m': sells_5m,
            'buyers_green': buyers_green,
            'buyers_blue': buyers_blue,
            'buyers_yellow': buyers_yellow,
            'buyers_red': buyers_red,
            'buyers_clown': buyers_clown,
            'buyers_sun': buyers_sun,
            'buyers_moon_half': buyers_moon_half,
            'buyers_moon_new': buyers_moon_new,
            'current_ratio': current_ratio,
            'initial_ratio': initial_ratio,
            'total_holders': total_holders,
            'freshies_1d_percent': freshies_1d_percent,
            'freshies_7d_percent': freshies_7d_percent,
            'top_10_percent': top_10_percent,
            'top_10_holdings': top_10_holdings,
            'dev_current_balance_percent': dev_current_balance_percent,
            'dev_sol_balance': dev_sol_balance,
            'security_no_mint': security_no_mint,
            'security_blacklist': security_blacklist,
            'security_burnt': security_burnt,
            'security_dev_sold': security_dev_sold,
            'security_dex_paid': security_dex_paid
        }
        
        logger.info(f"✅ Обработка данных успешна для {symbol}: MC=${market_cap:,.0f}, Liq=${liquidity:,.0f}")
        return processed_data
        
    except Exception as e:
        logger.error(f"❌ Ошибка обработки данных: {e}")
        raise ValueError(f"Ошибка обработки данных токена: {e}")

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
            'predict_batch': '/predict-batch [POST]',
            'health': '/health [GET]',
            'model_info': '/model-info [GET]'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Основной endpoint для предсказания одного токена"""
    
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
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Если данные пришли в виде массива, берем первый элемент
        if isinstance(data, list):
            if len(data) == 0:
                return jsonify({
                    'success': False,
                    'error': 'Empty array provided'
                }), 400
            token_json = data[0]
        else:
            token_json = data
        
        # Обрабатываем JSON данные токена
        processed_token_data = process_token_data(token_json)
        
        # Делаем предсказание
        prediction_result = predict_token_success(processed_token_data)
        
        # Анализируем сигналы
        signals = analyze_token_signals(processed_token_data)
        
        # Проверяем безопасность
        safety = check_token_safety(processed_token_data)
        
        # Получаем рекомендацию
        recommendation = get_recommendation(prediction_result, signals, safety)
        
        # Формируем ответ
        response = {
            'success': True,
            'token_symbol': processed_token_data.get('symbol', 'UNKNOWN'),
            'processed_data': processed_token_data,
            'prediction': prediction_result,
            'signals': signals,
            'safety': safety,
            'recommendation': recommendation,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"✅ Предсказание: {processed_token_data.get('symbol', 'UNKNOWN')} -> {prediction_result['prediction']} ({prediction_result['probability_percent']}%)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict-batch', methods=['POST'])
def predict_batch():
    """Endpoint для пакетного предсказания нескольких токенов"""
    
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
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Должен быть массив токенов
        if not isinstance(data, list):
            return jsonify({
                'success': False,
                'error': 'Expected array of tokens'
            }), 400
        
        results = []
        
        for i, token_json in enumerate(data):
            try:
                # Обрабатываем каждый токен
                processed_token_data = process_token_data(token_json)
                prediction_result = predict_token_success(processed_token_data)
                signals = analyze_token_signals(processed_token_data)
                safety = check_token_safety(processed_token_data)
                recommendation = get_recommendation(prediction_result, signals, safety)
                
                token_result = {
                    'index': i,
                    'success': True,
                    'token_symbol': processed_token_data.get('symbol', 'UNKNOWN'),
                    'prediction': prediction_result,
                    'signals': signals,
                    'safety': safety,
                    'recommendation': recommendation
                }
                
                results.append(token_result)
                
            except Exception as e:
                logger.error(f"❌ Ошибка обработки токена {i}: {e}")
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e),
                    'token_symbol': token_json.get('symbol', 'UNKNOWN')
                })
        
        # Статистика обработки
        successful = sum(1 for r in results if r.get('success'))
        failed = len(results) - successful
        
        response = {
            'success': True,
            'total_tokens': len(data),
            'successful_predictions': successful,
            'failed_predictions': failed,
            'results': results,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"✅ Пакетное предсказание: {successful}/{len(data)} успешно")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка пакетного API: {e}")
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
    """Пример входных данных в структурированном JSON формате"""
    
    example_data = {
        "symbol": "CRUMB",
        "name": "Crumbcat",
        "contract_address": "Hz7MeU72BNF9rCWyUFAwKTyCcjr6qsJm1jwYehnqjups",
        "token_age": "2m",
        "views": 59,
        "market_cap": "129.1K",
        "liquidity": "47.3K",
        "sol_pooled": None,
        "ath": "152.7K",
        "ath_change_percent": -21,
        "ath_time_ago": "32s",
        "volume_1m": 148947.33,
        "buy_volume_1m": 81396.06,
        "sell_volume_1m": 67551.28,
        "buys_1m": 567,
        "sells_1m": 453,
        "volume_5m": 172794.94,
        "buy_volume_5m": 98494.49,
        "sell_volume_5m": 74300.45,
        "buys_5m": 670,
        "sells_5m": 517,
        "first_buyers": {
            "visual_map": "⭕⭕⭕⭕🟡⭕🔵⭕⭕🟡🟡🟡⭕🟡⭕🟡⭕⭕⭕⭕⭕🟡🟡⭕🟢🟡🟡⭕⭕⭕🟡🟡🟢🟢⭕🟡🟡🟢🟡🟡🟡⭕🟡🟡🟢🟢🟢⭕⭕🟡🟢🟡⭕🟡🟡🟢🟡🔵🟡⭕⭕🟢🟡🟢⭕⭕🟢🟢🟢🟡",
            "green": 14,
            "blue": 2,
            "yellow": 27,
            "red": 27,
            "clown": 0,
            "sun": 0,
            "moon_half": 0,
            "moon_new": 0
        },
        "current_initial_ratio": {
            "current": 23.8,
            "initial": 72.58
        },
        "total_holders": 383,
        "freshies_1d_percent": 5.5,
        "freshies_7d_percent": 18,
        "top_10_percent": 21,
        "top_10_holdings": [29.2, 3.38, 3.32, 2.96, 2.95, 2.56, 2.49, 2.26, 1.95, 1.89],
        "dev_current_balance_percent": 0,
        "dev_sol_balance": 0.225,
        "security": {
            "no_mint": True,
            "blacklist": True,
            "burnt": True,
            "dev_sold": True,
            "dex_paid": False
        }
    }
    
    return jsonify({
        'single_token_request': {
            'url': request.base_url.replace('/example', '/predict'),
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': example_data
        },
        'array_request': {
            'url': request.base_url.replace('/example', '/predict'),
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': [example_data]
        },
        'batch_request': {
            'url': request.base_url.replace('/example', '/predict-batch'),
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': [example_data, example_data]
        }
    })

# Обработка ошибок
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/predict', '/predict-batch', '/health', '/model-info', '/example']
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
import os
import pickle
import pandas as pd
import numpy as np
import re
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

def parse_token_data(cena_full_text):
    """Парсит текстовые данные токена в структурированный формат"""
    
    try:
        # Извлекаем символ токена
        symbol_match = re.search(r'\$([A-Z0-9]+)', cena_full_text)
        symbol = symbol_match.group(1) if symbol_match else "UNKNOWN"
        
        # Извлекаем возраст токена (в минутах)
        age_match = re.search(r'Token age:\s*(\d+)m', cena_full_text)
        token_age_minutes = int(age_match.group(1)) if age_match else 0
        
        # Извлекаем рыночные данные
        mc_match = re.search(r'MC: \$([0-9,\.]+)K', cena_full_text)
        market_cap = float(mc_match.group(1).replace(',', '')) * 1000 if mc_match else 0
        
        liq_match = re.search(r'Liq: \$([0-9,\.]+)K', cena_full_text)
        liquidity = float(liq_match.group(1).replace(',', '')) * 1000 if liq_match else 0
        
        # SOL pooled
        sol_match = re.search(r'SOL pooled: ([0-9,\.]+)', cena_full_text)
        sol_pooled = float(sol_match.group(1).replace(',', '')) if sol_match else 0
        
        # ATH
        ath_match = re.search(r'ATH: \$([0-9,\.]+)K', cena_full_text)
        ath = float(ath_match.group(1).replace(',', '')) * 1000 if ath_match else 0
        
        # Объемы 1 минута
        vol_1m_match = re.search(r'Volume: \$([0-9,\.]+)', cena_full_text)
        volume_1m = float(vol_1m_match.group(1).replace(',', '')) if vol_1m_match else 0
        
        buy_vol_1m_match = re.search(r'Buy volume \(\$\): \$([0-9,\.]+)', cena_full_text)
        buy_volume_1m = float(buy_vol_1m_match.group(1).replace(',', '')) if buy_vol_1m_match else 0
        
        sell_vol_1m_match = re.search(r'Sell volume \(\$\): \$([0-9,\.]+)', cena_full_text)
        sell_volume_1m = float(sell_vol_1m_match.group(1).replace(',', '')) if sell_vol_1m_match else 0
        
        # Количество покупок/продаж
        buys_1m_match = re.search(r'Buys: (\d+)', cena_full_text)
        buys_1m = int(buys_1m_match.group(1)) if buys_1m_match else 0
        
        sells_1m_match = re.search(r'Sells: (\d+)', cena_full_text)
        sells_1m = int(sells_1m_match.group(1)) if sells_1m_match else 0
        
        # Объемы 5 минут (ищем вторые значения)
        vol_5m_matches = re.findall(r'Volume: \$([0-9,\.]+)', cena_full_text)
        volume_5m = float(vol_5m_matches[1].replace(',', '')) if len(vol_5m_matches) > 1 else 0
        
        buy_vol_5m_matches = re.findall(r'Buy volume \(\$\): \$([0-9,\.]+)', cena_full_text)
        buy_volume_5m = float(buy_vol_5m_matches[1].replace(',', '')) if len(buy_vol_5m_matches) > 1 else 0
        
        sell_vol_5m_matches = re.findall(r'Sell volume \(\$\): \$([0-9,\.]+)', cena_full_text)
        sell_volume_5m = float(sell_vol_5m_matches[1].replace(',', '')) if len(sell_vol_5m_matches) > 1 else 0
        
        buys_5m_matches = re.findall(r'Buys: (\d+)', cena_full_text)
        buys_5m = int(buys_5m_matches[1]) if len(buys_5m_matches) > 1 else 0
        
        sells_5m_matches = re.findall(r'Sells: (\d+)', cena_full_text)
        sells_5m = int(sells_5m_matches[1]) if len(sells_5m_matches) > 1 else 0
        
        # Покупатели по категориям
        green_match = re.search(r'🟢: (\d+)', cena_full_text)
        buyers_green = int(green_match.group(1)) if green_match else 0
        
        blue_match = re.search(r'🔵: (\d+)', cena_full_text)
        buyers_blue = int(blue_match.group(1)) if blue_match else 0
        
        yellow_match = re.search(r'🟡: (\d+)', cena_full_text)
        buyers_yellow = int(yellow_match.group(1)) if yellow_match else 0
        
        red_match = re.search(r'⭕️: (\d+)', cena_full_text)
        buyers_red = int(red_match.group(1)) if red_match else 0
        
        # Снайперы
        clown_match = re.search(r'🤡: (\d+)', cena_full_text)
        buyers_clown = int(clown_match.group(1)) if clown_match else 0
        
        sun_match = re.search(r'🌞: (\d+)', cena_full_text)
        buyers_sun = int(sun_match.group(1)) if sun_match else 0
        
        moon_half_match = re.search(r'🌗: (\d+)', cena_full_text)
        buyers_moon_half = int(moon_half_match.group(1)) if moon_half_match else 0
        
        moon_new_match = re.search(r'🌚: (\d+)', cena_full_text)
        buyers_moon_new = int(moon_new_match.group(1)) if moon_new_match else 0
        
        # Current/Initial ratio
        ratio_match = re.search(r'Current/Initial: ([0-9\.]+)% / ([0-9\.]+)%', cena_full_text)
        current_ratio = float(ratio_match.group(1)) if ratio_match else 0
        initial_ratio = float(ratio_match.group(2)) if ratio_match else 0
        
        # Держатели
        total_holders_match = re.search(r'Total: (\d+)', cena_full_text)
        total_holders = int(total_holders_match.group(1)) if total_holders_match else 0
        
        freshies_1d_match = re.search(r'Freshies: ([0-9\.]+)% 1D', cena_full_text)
        freshies_1d_percent = float(freshies_1d_match.group(1)) if freshies_1d_match else 0
        
        freshies_7d_match = re.search(r'([0-9\.]+)% 7D', cena_full_text)
        freshies_7d_percent = float(freshies_7d_match.group(1)) if freshies_7d_match else 0
        
        top_10_match = re.search(r'Top 10: (\d+)%', cena_full_text)
        top_10_percent = float(top_10_match.group(1)) if top_10_match else 0
        
        # Top 10 holdings - берем первое значение
        holdings_match = re.search(r'Top 10 Holding.*?\n([0-9\.]+)', cena_full_text, re.DOTALL)
        top_10_holdings = float(holdings_match.group(1)) if holdings_match else 0
        
        # Dev данные
        dev_balance_match = re.search(r'Dev current balance: (\d+)%', cena_full_text)
        dev_current_balance_percent = float(dev_balance_match.group(1)) if dev_balance_match else 0
        
        dev_sol_match = re.search(r'Dev SOL balance: ([0-9\.]+) SOL', cena_full_text)
        dev_sol_balance = float(dev_sol_match.group(1)) if dev_sol_match else 0
        
        # Security flags (🟢 = 1, 🔴 = 0)
        security_no_mint = 1 if '├ NoMint: 🟢' in cena_full_text else 0
        security_blacklist = 1 if '├ Blacklist: 🟢' in cena_full_text else 0
        security_burnt = 1 if '├ Burnt: 🟢' in cena_full_text else 0
        security_dev_sold = 1 if '├ Dev Sold: 🟢' in cena_full_text else 0
        security_dex_paid = 1 if '└ Dex Paid: 🟢' in cena_full_text else 0
        
        # Формируем структурированные данные
        parsed_data = {
            'symbol': symbol,
            'token_age_minutes': token_age_minutes,
            'market_cap': market_cap,
            'liquidity': liquidity,
            'sol_pooled': sol_pooled,
            'ath': ath,
            'volume_1m': volume_1m,
            'buy_volume_1m': buy_volume_1m,
            'sell_volume_1m': sell_volume_1m,
            'buys_1m': buys_1m,
            'sells_1m': sells_1m,
            'volume_5m': volume_5m,
            'buy_volume_5m': buy_volume_5m,
            'sell_volume_5m': sell_volume_5m,
            'buys_5m': buys_5m,
            'sells_5m': sells_5m,
            'buyers_green': buyers_green,
            'buyers_blue': buyers_blue,
            'buyers_yellow': buyers_yellow,
            'buyers_red': buyers_red,
            'buyers_clown': buyers_clown,
            'buyers_sun': buyers_sun,
            'buyers_moon_half': buyers_moon_half,
            'buyers_moon_new': buyers_moon_new,
            'current_ratio': current_ratio,
            'initial_ratio': initial_ratio,
            'total_holders': total_holders,
            'freshies_1d_percent': freshies_1d_percent,
            'freshies_7d_percent': freshies_7d_percent,
            'top_10_percent': top_10_percent,
            'top_10_holdings': top_10_holdings,
            'dev_current_balance_percent': dev_current_balance_percent,
            'dev_sol_balance': dev_sol_balance,
            'security_no_mint': security_no_mint,
            'security_blacklist': security_blacklist,
            'security_burnt': security_burnt,
            'security_dev_sold': security_dev_sold,
            'security_dex_paid': security_dex_paid
        }
        
        logger.info(f"✅ Парсинг успешен для {symbol}: MC=${market_cap:,.0f}, Liq=${liquidity:,.0f}")
        return parsed_data
        
    except Exception as e:
        logger.error(f"❌ Ошибка парсинга: {e}")
        raise ValueError(f"Ошибка парсинга данных токена: {e}")

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
        
        # Получаем порт из переменной окружения (Railway)
    port = int(os.environ.get('PORT', 5000))
    
    # Запускаем приложение
    app.run(host='0.0.0.0', port=port, debug=False)аем предсказания
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
            'predict_text': '/predict-text [POST]',
            'health': '/health [GET]',
            'model_info': '/model-info [GET]'
        }
    })

@app.route('/predict-text', methods=['POST'])
def predict_text():
    """Endpoint для предсказания из текстовых данных (основной для n8n)"""
    
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
        
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Ожидаем поле cena_full с текстовыми данными
        cena_full_text = data.get('cena_full')
        if not cena_full_text:
            return jsonify({
                'success': False,
                'error': 'Field "cena_full" is required'
            }), 400
        
        # Парсим текстовые данные
        parsed_token_data = parse_token_data(cena_full_text)
        
        # Делаем предсказание
        prediction_result = predict_token_success(parsed_token_data)
        
        # Анализируем сигналы
        signals = analyze_token_signals(parsed_token_data)
        
        # Проверяем безопасность
        safety = check_token_safety(parsed_token_data)
        
        # Получаем рекомендацию
        recommendation = get_recommendation(prediction_result, signals, safety)
        
        # Формируем ответ
        response = {
            'success': True,
            'token_symbol': parsed_token_data.get('symbol', 'UNKNOWN'),
            'parsed_data': parsed_token_data,
            'prediction': prediction_result,
            'signals': signals,
            'safety': safety,
            'recommendation': recommendation,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"✅ Предсказание: {parsed_token_data.get('symbol', 'UNKNOWN')} -> {prediction_result['prediction']} ({prediction_result['probability_percent']}%)")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"❌ Ошибка API: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint для предсказания из структурированных данных"""
    
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

@app.route('/example-text', methods=['GET'])
def example_text():
    """Пример входных данных в текстовом формате"""
    
    example_data = {
        "cena_full": "🎲 $CRUMB | Crumbcat\nHz7MeU72BNF9rCWyUFAwKTyCcjr6qsJm1jwYehnqjups\n⏳ Token age:  2m  | 👁 59\n├ MC: $129.1K\n├ Liq: $47.3K / SOL pooled: --\n└ ATH: $152.7K (-21% / 32s)\n1 min:\n├ Volume: $148,947.33\n├ Buy volume ($): $81,396.06\n├ Sell volume ($): $67,551.28\n├ Buys: 567\n└ Sells: 453\n5 min:\n├ Volume: $172,794.94\n├ Buy volume ($): $98,494.49\n├ Sell volume ($): $74,300.45\n├ Buys: 670\n└ Sells: 517\n🎯 First 70 buyers:\n⭕️⭕️⭕️⭕️🟡⭕️🔵⭕️⭕️🟡\n🟡🟡⭕️🟡⭕️🟡⭕️⭕️⭕️⭕️\n⭕️🟡🟡⭕️🟢🟡🟡⭕️⭕️⭕️\n🟡🟡🟢🟢⭕️🟡🟡🟢🟡🟡\n🟡⭕️🟡🟡🟢🟢🟢⭕️⭕️🟡\n🟢🟡⭕️🟡🟡🟢🟡🔵🟡⭕️\n⭕️🟢🟡🟢⭕️⭕️🟢🟢🟢🟡\n├ 🟢: 14 | 🔵: 2 | 🟡: 27 | ⭕️: 27\n├ 🤡: 0 | 🌞: 0 | 🌗: 0 | 🌚: 0\n├ Current/Initial: 23.8% / 72.58%\n👥 Holders:\n├ Total: 383\n├ Freshies: 5.5% 1D | 18% 7D\n├ Top 10: 21%\n💰 Top 10 Holding (%)\n29.2 | 3.38 | 3.32 | 2.96 | 2.95 | 2.56 | 2.49 | 2.26 | 1.95 | 1.89\n😎 Dev\n├ Dev current balance: 0%\n└ Dev SOL balance: 0.225 SOL\n🔒 Security:\n├ NoMint: 🟢\n├ Blacklist: 🟢\n├ Burnt: 🟢\n├ Dev Sold: 🟢\n└ Dex Paid: 🔴"
    }
    
    return jsonify({
        'example_request': {
            'url': request.base_url.replace('/example-text', '/predict-text'),
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': example_data
        },
        'required_fields': [
            'cena_full (string with full token data)'
        ],
        'note': 'Send the complete text data in cena_full field'
    })

# Обработка ошибок
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found',
        'available_endpoints': ['/predict-text', '/predict', '/health', '/model-info', '/example-text']
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
    
    # Получ# app.py - Простой API для Railway + n8n
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
