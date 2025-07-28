from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
import os

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

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

def parse_top10_holdings(holdings_list, total_top10_percent=None):
    """Обработка концентрации китов"""
    if not holdings_list or len(holdings_list) == 0:
        return {
            'top1_real_percent': 0, 'top3_real_percent': 0, 'top5_real_percent': 0,
            'concentration_ratio': 0, 'gini_coefficient': 0, 'herfindahl_index': 0
        }
    
    try:
        # Если это список чисел
        if isinstance(holdings_list, list):
            internal_percentages = [float(x) for x in holdings_list if x is not None]
        else:
            # Если это строка
            value_clean = str(holdings_list).strip('[]').replace(' ', '')
            internal_percentages = [float(x) for x in value_clean.split(',') if x.strip()]
        
        while len(internal_percentages) < 10:
            internal_percentages.append(0)
        
        if total_top10_percent is None or pd.isna(total_top10_percent) or total_top10_percent <= 0:
            real_percentages = internal_percentages
        else:
            total_internal = sum(internal_percentages)
            if total_internal > 0:
                normalized_percentages = [x / total_internal for x in internal_percentages]
                real_percentages = [x * total_top10_percent / 100 for x in normalized_percentages]
            else:
                real_percentages = [0] * 10
        
        # Основные метрики
        top1_real = real_percentages[0] if len(real_percentages) > 0 else 0
        top3_real = sum(real_percentages[:3]) if len(real_percentages) >= 3 else sum(real_percentages)
        top5_real = sum(real_percentages[:5]) if len(real_percentages) >= 5 else sum(real_percentages)
        
        # Коэффициент концентрации
        total_internal_nonzero = sum([x for x in internal_percentages if x > 0])
        concentration_ratio = internal_percentages[0] / total_internal_nonzero if total_internal_nonzero > 0 else 0
        
        # Коэффициент Джини
        sorted_percentages = sorted([x for x in internal_percentages if x > 0], reverse=True)
        n = len(sorted_percentages)
        if n > 1:
            cumsum = np.cumsum(sorted_percentages)
            gini_coefficient = (n + 1 - 2 * sum((n + 1 - i) * x for i, x in enumerate(cumsum))) / (n * sum(sorted_percentages))
        else:
            gini_coefficient = 0
            
        # Индекс Херфиндаля
        total_sum = sum(internal_percentages)
        if total_sum > 0:
            herfindahl_index = sum((x / total_sum) ** 2 for x in internal_percentages if x > 0)
        else:
            herfindahl_index = 0
        
        return {
            'top1_real_percent': top1_real,
            'top3_real_percent': top3_real,
            'top5_real_percent': top5_real,
            'concentration_ratio': concentration_ratio,
            'gini_coefficient': gini_coefficient,
            'herfindahl_index': herfindahl_index
        }
    except Exception as e:
        logger.error(f"Error parsing holdings: {e}")
        return {
            'top1_real_percent': 0, 'top3_real_percent': 0, 'top5_real_percent': 0,
            'concentration_ratio': 0, 'gini_coefficient': 0, 'herfindahl_index': 0
        }

def convert_input_to_features(token_data):
    """Конвертирует входные данные в признаки для модели"""
    try:
        # Создаем DataFrame
        df = pd.DataFrame([token_data])
        
        # Базовая обработка рыночных данных
        df['market_cap_numeric'] = df['market_cap'].apply(parse_string_number)
        df['liquidity_numeric'] = df['liquidity'].apply(parse_string_number)
        df['ath_numeric'] = df['ath'].apply(parse_string_number)
        
        # Обработка выбросов (ограничиваем на разумных уровнях)
        df['market_cap_capped'] = df['market_cap_numeric'].clip(upper=10_000_000)  # 10M cap
        df['liquidity_capped'] = df['liquidity_numeric'].clip(upper=1_000_000)    # 1M liquidity
        df['ath_capped'] = df['ath_numeric'].clip(upper=1_000_000)                # 1M ATH
        
        # Обработка времени
        df['token_age_minutes'] = df['token_age'].apply(parse_time_to_minutes)
        df['token_age_hours'] = df['token_age_minutes'] / 60
        df['token_age_days'] = df['token_age_minutes'] / 1440
        
        # Категориальные признаки времени
        df['is_very_new'] = (df['token_age_minutes'] < 60).astype(int)
        df['is_new'] = (df['token_age_minutes'] < 1440).astype(int)
        df['is_mature'] = (df['token_age_minutes'] > 10080).astype(int)
        df['is_very_mature'] = (df['token_age_minutes'] > 43200).astype(int)
        
        # Логарифмические преобразования времени
        df['token_age_log'] = np.log1p(df['token_age_minutes'])
        df['token_age_sqrt'] = np.sqrt(df['token_age_minutes'])
        
        # Торговые паттерны
        df['total_volume_1m'] = df['buy_volume_1m'] + df['sell_volume_1m']
        df['total_volume_5m'] = df['buy_volume_5m'] + df['sell_volume_5m']
        
        df['buy_sell_ratio_1m'] = np.where(df['sell_volume_1m'] > 0, 
                                          df['buy_volume_1m'] / df['sell_volume_1m'], 0)
        df['buy_sell_ratio_5m'] = np.where(df['sell_volume_5m'] > 0, 
                                          df['buy_volume_5m'] / df['sell_volume_5m'], 0)
        
        df['buy_pressure_1m'] = np.where(df['total_volume_1m'] > 0, 
                                        df['buy_volume_1m'] / df['total_volume_1m'], 0)
        df['buy_pressure_5m'] = np.where(df['total_volume_5m'] > 0, 
                                        df['buy_volume_5m'] / df['total_volume_5m'], 0)
        df['buy_pressure_change'] = df['buy_pressure_5m'] - df['buy_pressure_1m']
        
        # Средние размеры сделок
        df['avg_buy_size_1m'] = np.where(df['buys_1m'] > 0, 
                                        df['buy_volume_1m'] / df['buys_1m'], 0)
        df['avg_sell_size_1m'] = np.where(df['sells_1m'] > 0, 
                                         df['sell_volume_1m'] / df['sells_1m'], 0)
        
        # Поведенческие паттерны из first_buyers
        first_buyers = token_data.get('first_buyers', {})
        df['buyers_green'] = first_buyers.get('green', 0)
        df['buyers_blue'] = first_buyers.get('blue', 0)
        df['buyers_yellow'] = first_buyers.get('yellow', 0)
        df['buyers_red'] = first_buyers.get('red', 0)
        df['buyers_clown'] = first_buyers.get('clown', 0)
        df['buyers_sun'] = first_buyers.get('sun', 0)
        df['buyers_moon_half'] = first_buyers.get('moon_half', 0)
        df['buyers_moon_new'] = first_buyers.get('moon_new', 0)
        
        df['total_holders_emoji'] = (df['buyers_green'] + df['buyers_blue'] + 
                                    df['buyers_yellow'] + df['buyers_red'])
        df['total_snipers'] = (df['buyers_clown'] + df['buyers_sun'] + 
                              df['buyers_moon_half'] + df['buyers_moon_new'])
        
        df['holders_diamond_hands'] = np.where(df['total_holders_emoji'] > 0,
                                              df['buyers_green'] / df['total_holders_emoji'], 0)
        df['holders_paper_hands'] = np.where(df['total_holders_emoji'] > 0,
                                            df['buyers_red'] / df['total_holders_emoji'], 0)
        
        df['trust_score'] = (df['buyers_green'] + df['buyers_clown']) / (df['total_holders_emoji'] + df['total_snipers'] + 1)
        df['distrust_score'] = (df['buyers_red'] + df['buyers_moon_new']) / (df['total_holders_emoji'] + df['total_snipers'] + 1)
        
        # Рыночные коэффициенты
        df['liquidity_to_mcap_ratio'] = np.where(df['market_cap_capped'] > 0,
                                                df['liquidity_capped'] / df['market_cap_capped'], 0)
        df['volume_to_liquidity_ratio'] = np.where(df['liquidity_capped'] > 0,
                                                  df['total_volume_5m'] / df['liquidity_capped'], 0)
        df['volume_to_mcap_ratio'] = np.where(df['market_cap_capped'] > 0,
                                             df['total_volume_5m'] / df['market_cap_capped'], 0)
        
        df['volume_per_holder'] = np.where(df['total_holders'] > 0,
                                          df['total_volume_5m'] / df['total_holders'], 0)
        
        # Обработка концентрации китов
        top_10_holdings = token_data.get('top_10_holdings', [])
        top_10_percent = token_data.get('top_10_percent', 0)
        
        whale_metrics = parse_top10_holdings(top_10_holdings, top_10_percent)
        
        df['biggest_whale_percent'] = whale_metrics['top1_real_percent']
        df['top3_whales_percent'] = whale_metrics['top3_real_percent']
        df['top5_whales_percent'] = whale_metrics['top5_real_percent']
        df['whale_dominance_index'] = whale_metrics['concentration_ratio']
        df['gini_coefficient'] = whale_metrics['gini_coefficient']
        df['herfindahl_index'] = whale_metrics['herfindahl_index']
        df['whale_centralization'] = whale_metrics['top1_real_percent'] / 100.0
        df['dangerous_whale_concentration'] = int(whale_metrics['top1_real_percent'] > 15)
        df['safe_whale_concentration'] = int(whale_metrics['top1_real_percent'] <= 5)
        
        # Логарифмические преобразования
        log_features = ['market_cap_capped', 'liquidity_capped', 'ath_capped', 'total_holders', 'total_volume_5m']
        for col in log_features:
            if col in df.columns:
                df[f'{col}_log'] = np.log1p(df[col])
                df[f'{col}_sqrt'] = np.sqrt(df[col])
                df[f'{col}_inv'] = 1 / (df[col] + 1)
        
        # Взаимодействия признаков
        df['volume_liquidity_interaction'] = df['total_volume_5m'] * df.get('liquidity_capped_log', 0)
        df['volume_mcap_interaction'] = df['total_volume_5m'] * df.get('market_cap_capped_log', 0)
        df['age_volume_interaction'] = df.get('token_age_log', 0) * df['total_volume_5m']
        df['trust_whale_interaction'] = df['trust_score'] * df.get('whale_centralization', 0)
        
        # Безопасность
        security = token_data.get('security', {})
        df['security_no_mint'] = int(security.get('no_mint', False))
        df['security_blacklist'] = int(security.get('blacklist', False))
        df['security_burnt'] = int(security.get('burnt', False))
        df['security_dev_sold'] = int(security.get('dev_sold', False))
        df['security_dex_paid'] = int(security.get('dex_paid', False))
        
        security_features = ['security_no_mint', 'security_blacklist', 'security_burnt', 
                            'security_dev_sold', 'security_dex_paid']
        df['security_score'] = df[security_features].sum(axis=1)
        df['security_perfect'] = (df['security_score'] == 5).astype(int)
        
        # Текущий/начальный рейтинг
        current_initial = token_data.get('current_initial_ratio', {})
        df['current_ratio'] = current_initial.get('current', 0)
        df['initial_ratio'] = current_initial.get('initial', 0)
        df['ratio_change'] = df['current_ratio'] - df['initial_ratio']
        df['ratio_change_percent'] = np.where(df['initial_ratio'] > 0,
                                             (df['current_ratio'] - df['initial_ratio']) / df['initial_ratio'], 0)
        
        # Дополнительные признаки
        df['sol_pooled'] = token_data.get('sol_pooled', 0)
        df['freshies_1d_percent'] = token_data.get('freshies_1d_percent', 0)
        df['freshies_7d_percent'] = token_data.get('freshies_7d_percent', 0)
        df['dev_current_balance_percent'] = token_data.get('dev_current_balance_percent', 0)
        
        # Заполняем пропуски
        df = df.fillna(0)
        
        return df.iloc[0].to_dict()
        
    except Exception as e:
        logger.error(f"Error converting features: {e}")
        raise

# =============================================================================
# ПРОСТАЯ МОДЕЛЬ (заглушка для демонстрации)
# =============================================================================

def simple_predict(features):
    """
    Простая модель предсказания (заглушка)
    В реальном приложении здесь будет загруженная обученная модель
    """
    try:
        # Простые правила для демонстрации
        score = 0.5  # Базовая вероятность
        
        # Возраст токена
        age_minutes = features.get('token_age_minutes', 0)
        if age_minutes < 60:  # Очень новые токены рискованнее
            score -= 0.1
        elif 60 <= age_minutes <= 1440:  # Оптимальный возраст
            score += 0.1
        
        # Давление покупок
        buy_pressure = features.get('buy_pressure_5m', 0)
        if buy_pressure > 0.6:  # Сильное давление покупок
            score += 0.15
        elif buy_pressure < 0.4:  # Слабое давление покупок
            score -= 0.15
        
        # Концентрация китов
        whale_percent = features.get('biggest_whale_percent', 0)
        if whale_percent > 20:  # Опасная концентрация
            score -= 0.2
        elif whale_percent < 10:  # Безопасная концентрация
            score += 0.1
        
        # Trust score
        trust = features.get('trust_score', 0)
        if trust > 0.6:  # Высокое доверие
            score += 0.1
        elif trust < 0.3:  # Низкое доверие
            score -= 0.1
        
        # Ликвидность к market cap
        liq_mcap_ratio = features.get('liquidity_to_mcap_ratio', 0)
        if liq_mcap_ratio > 0.5:  # Хорошая ликвидность
            score += 0.05
        elif liq_mcap_ratio < 0.2:  # Плохая ликвидность
            score -= 0.1
        
        # Безопасность
        security_score = features.get('security_score', 0)
        if security_score >= 4:  # Высокая безопасность
            score += 0.1
        elif security_score <= 2:  # Низкая безопасность
            score -= 0.15
        
        # Ограничиваем вероятность
        probability = max(0.0, min(1.0, score))
        
        # Предсказание
        prediction = 1 if probability > 0.5 else 0
        
        # Рекомендация
        if probability >= 0.85:
            recommendation = "🔥 VERY STRONG BUY"
        elif probability >= 0.75:
            recommendation = "🚀 STRONG BUY" 
        elif probability >= 0.65:
            recommendation = "✅ BUY"
        elif probability >= 0.55:
            recommendation = "⚖️ CONSIDER"
        elif probability >= 0.45:
            recommendation = "⚠️ CAUTION"
        elif probability >= 0.35:
            recommendation = "❌ AVOID"
        else:
            recommendation = "🚫 STRONG AVOID"
        
        confidence_interval = (max(0, probability - 0.1), min(1, probability + 0.1))
        
        return {
            'probability': float(probability),
            'prediction': int(prediction),
            'recommendation': recommendation,
            'confidence_interval': confidence_interval,
            'key_factors': {
                'age_minutes': age_minutes,
                'buy_pressure_5m': buy_pressure,
                'biggest_whale_percent': whale_percent,
                'trust_score': trust,
                'security_score': security_score,
                'liquidity_to_mcap_ratio': liq_mcap_ratio
            }
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        return {
            'probability': 0.5,
            'prediction': 0,
            'recommendation': "❓ ERROR",
            'confidence_interval': (0.0, 1.0),
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
        'version': '1.0.0'
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
        
        # Получаем предсказание
        result = simple_predict(features)
        
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
    """Пакетное предсказание для нескольких токенов"""
    try:
        data = request.get_json()
        
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Data should be a list of token objects'}), 400
        
        if len(data) > 100:  # Ограничиваем размер пакета
            return jsonify({'error': 'Batch size cannot exceed 100 tokens'}), 400
        
        logger.info(f"Received batch prediction request for {len(data)} tokens")
        
        results = []
        
        for i, token_data in enumerate(data):
            try:
                # Конвертируем входные данные в признаки
                features = convert_input_to_features(token_data)
                
                # Получаем предсказание
                result = simple_predict(features)
                
                # Добавляем информацию о токене
                result['token_info'] = {
                    'symbol': token_data.get('symbol', ''),
                    'name': token_data.get('name', ''),
                    'contract_address': token_data.get('contract_address', ''),
                    'market_cap': token_data.get('market_cap', ''),
                    'liquidity': token_data.get('liquidity', ''),
                    'token_age': token_data.get('token_age', ''),
                    'batch_index': i
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing token {i}: {e}")
                results.append({
                    'error': f'Error processing token {i}',
                    'message': str(e),
                    'token_info': {
                        'symbol': token_data.get('symbol', ''),
                        'batch_index': i
                    }
                })
        
        # Статистика пакета
        successful_predictions = [r for r in results if 'error' not in r]
        avg_probability = sum(r['probability'] for r in successful_predictions) / len(successful_predictions) if successful_predictions else 0
        
        response = {
            'results': results,
            'batch_stats': {
                'total_tokens': len(data),
                'successful_predictions': len(successful_predictions),
                'failed_predictions': len(results) - len(successful_predictions),
                'average_probability': avg_probability,
                'timestamp': datetime.now().isoformat()
            }
        }
        
        logger.info(f"Batch prediction completed: {len(successful_predictions)}/{len(data)} successful")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in batch predict endpoint: {e}")
        return jsonify({
            'error': 'Internal server error', 
            'message': str(e)
        }), 500

@app.route('/analyze', methods=['POST'])
def analyze():
    """Детальный анализ токена"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        logger.info(f"Received analysis request for token: {data.get('symbol', 'unknown')}")
        
        # Конвертируем входные данные в признаки
        features = convert_input_to_features(data)
        
        # Получаем предсказание
        prediction_result = simple_predict(features)
        
        # Детальный анализ
        analysis = {
            'token_info': {
                'symbol': data.get('symbol', ''),
                'name': data.get('name', ''),
                'contract_address': data.get('contract_address', ''),
                'timestamp': datetime.now().isoformat()
            },
            'prediction': prediction_result,
            'detailed_analysis': {
                'market_metrics': {
                    'market_cap': data.get('market_cap', ''),
                    'liquidity': data.get('liquidity', ''),
                    'sol_pooled': data.get('sol_pooled', 0),
                    'liquidity_to_mcap_ratio': features.get('liquidity_to_mcap_ratio', 0),
                    'assessment': 'Good' if features.get('liquidity_to_mcap_ratio', 0) > 0.3 else 'Poor'
                },
                'age_analysis': {
                    'token_age': data.get('token_age', ''),
                    'age_minutes': features.get('token_age_minutes', 0),
                    'category': 'Very New' if features.get('token_age_minutes', 0) < 60 else 
                               'New' if features.get('token_age_minutes', 0) < 1440 else 'Mature',
                    'risk_level': 'High' if features.get('token_age_minutes', 0) < 60 else 'Medium'
                },
                'trading_activity': {
                    'volume_5m': data.get('volume_5m', 0),
                    'buy_pressure_5m': features.get('buy_pressure_5m', 0),
                    'buy_sell_ratio_5m': features.get('buy_sell_ratio_5m', 0),
                    'activity_level': 'High' if data.get('volume_5m', 0) > 100000 else 
                                    'Medium' if data.get('volume_5m', 0) > 10000 else 'Low'
                },
                'whale_concentration': {
                    'biggest_whale_percent': features.get('biggest_whale_percent', 0),
                    'top3_whales_percent': features.get('top3_whales_percent', 0),
                    'risk_assessment': 'High Risk' if features.get('biggest_whale_percent', 0) > 20 else
                                     'Medium Risk' if features.get('biggest_whale_percent', 0) > 10 else 'Low Risk'
                },
                'holder_behavior': {
                    'trust_score': features.get('trust_score', 0),
                    'distrust_score': features.get('distrust_score', 0),
                    'diamond_hands_ratio': features.get('holders_diamond_hands', 0),
                    'paper_hands_ratio': features.get('holders_paper_hands', 0),
                    'sentiment': 'Positive' if features.get('trust_score', 0) > 0.6 else
                               'Negative' if features.get('trust_score', 0) < 0.3 else 'Neutral'
                },
                'security_analysis': {
                    'security_score': features.get('security_score', 0),
                    'no_mint': bool(features.get('security_no_mint', 0)),
                    'blacklist_protected': bool(features.get('security_blacklist', 0)),
                    'burnt': bool(features.get('security_burnt', 0)),
                    'dev_sold': bool(features.get('security_dev_sold', 0)),
                    'dex_paid': bool(features.get('security_dex_paid', 0)),
                    'overall_security': 'High' if features.get('security_score', 0) >= 4 else
                                      'Medium' if features.get('security_score', 0) >= 2 else 'Low'
                }
            },
            'risk_factors': [],
            'positive_factors': []
        }
        
        # Добавляем факторы риска
        if features.get('biggest_whale_percent', 0) > 20:
            analysis['risk_factors'].append('High whale concentration (>20%)')
        
        if features.get('token_age_minutes', 0) < 60:
            analysis['risk_factors'].append('Very new token (<1 hour)')
        
        if features.get('buy_pressure_5m', 0) < 0.4:
            analysis['risk_factors'].append('Low buying pressure')
        
        if features.get('security_score', 0) < 3:
            analysis['risk_factors'].append('Low security score')
        
        if features.get('liquidity_to_mcap_ratio', 0) < 0.2:
            analysis['risk_factors'].append('Low liquidity ratio')
        
        # Добавляем положительные факторы
        if features.get('buy_pressure_5m', 0) > 0.6:
            analysis['positive_factors'].append('Strong buying pressure')
        
        if features.get('trust_score', 0) > 0.6:
            analysis['positive_factors'].append('High holder trust')
        
        if features.get('security_score', 0) >= 4:
            analysis['positive_factors'].append('High security standards')
        
        if features.get('biggest_whale_percent', 0) < 10:
            analysis['positive_factors'].append('Low whale concentration')
        
        if 60 <= features.get('token_age_minutes', 0) <= 1440:
            analysis['positive_factors'].append('Optimal token age')
        
        logger.info(f"Analysis completed for {data.get('symbol', 'unknown')}")
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Error in analyze endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

@app.route('/features', methods=['POST'])
def extract_features():
    """Извлечение признаков из данных токена"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Конвертируем входные данные в признаки
        features = convert_input_to_features(data)
        
        # Группируем признаки по категориям
        categorized_features = {
            'basic_info': {
                'symbol': data.get('symbol', ''),
                'name': data.get('name', ''),
                'contract_address': data.get('contract_address', ''),
                'token_age': data.get('token_age', ''),
                'token_age_minutes': features.get('token_age_minutes', 0)
            },
            'market_data': {
                'market_cap': data.get('market_cap', ''),
                'market_cap_capped': features.get('market_cap_capped', 0),
                'liquidity': data.get('liquidity', ''),
                'liquidity_capped': features.get('liquidity_capped', 0),
                'sol_pooled': features.get('sol_pooled', 0),
                'liquidity_to_mcap_ratio': features.get('liquidity_to_mcap_ratio', 0)
            },
            'trading_metrics': {
                'volume_1m': data.get('volume_1m', 0),
                'volume_5m': data.get('volume_5m', 0),
                'buy_pressure_1m': features.get('buy_pressure_1m', 0),
                'buy_pressure_5m': features.get('buy_pressure_5m', 0),
                'buy_pressure_change': features.get('buy_pressure_change', 0),
                'buy_sell_ratio_5m': features.get('buy_sell_ratio_5m', 0)
            },
            'holder_behavior': {
                'total_holders': data.get('total_holders', 0),
                'trust_score': features.get('trust_score', 0),
                'distrust_score': features.get('distrust_score', 0),
                'holders_diamond_hands': features.get('holders_diamond_hands', 0),
                'holders_paper_hands': features.get('holders_paper_hands', 0)
            },
            'whale_analysis': {
                'biggest_whale_percent': features.get('biggest_whale_percent', 0),
                'top3_whales_percent': features.get('top3_whales_percent', 0),
                'top5_whales_percent': features.get('top5_whales_percent', 0),
                'whale_centralization': features.get('whale_centralization', 0),
                'gini_coefficient': features.get('gini_coefficient', 0)
            },
            'security': {
                'security_score': features.get('security_score', 0),
                'security_no_mint': bool(features.get('security_no_mint', 0)),
                'security_blacklist': bool(features.get('security_blacklist', 0)),
                'security_burnt': bool(features.get('security_burnt', 0)),
                'security_dev_sold': bool(features.get('security_dev_sold', 0)),
                'security_dex_paid': bool(features.get('security_dex_paid', 0))
            }
        }
        
        return jsonify({
            'categorized_features': categorized_features,
            'all_features': features,
            'feature_count': len(features),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in features endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': str(e)
        }), 500

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
    app.run(host='0.0.0.0', port=port, debug=os.environ.get('DEBUG', 'False').lower() == 'true')
