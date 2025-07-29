# app.py
import os
import pandas as pd
import numpy as np
import re
import joblib
import json # Для парсинга 'body'
import logging
from flask import Flask, request, jsonify
from datetime import datetime
# from sklearn.preprocessing import LabelEncoder # Не импортируем, так как используем сохраненный

# --- Настройка логгирования ---
# Уровень INFO для Railway, чтобы видеть важные сообщения в логах
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Инициализация Flask приложения ---
app = Flask(__name__)

# --- Глобальные переменные для модели и компонентов ---
model = None
scaler = None
encoders = None # В ноутбуке это label_encoders
features = None # Список признаков
model_name = None
ensemble_weights = None

# --- Функции обработки данных (скопированы из вашего ноутбука) ---

def parse_string_number(value):
    """Улучшенная версия парсинга строковых чисел"""
    # Преобразование массивов/серий в скаляр для надежности
    if isinstance(value, (list, np.ndarray, pd.Series)):
        if len(value) == 0:
            value = '' 
        else:
            value = value[0] if not isinstance(value, pd.Series) else (value.iloc[0] if len(value) > 0 else '')
    # Проверка на NaN, None, пустые строки
    if pd.isna(value) or value == '' or value == 'N/A' or value is None:
        return 0.0
    # Если уже число, возвращаем его
    if isinstance(value, (int, float)):
        return float(value)
    # Преобразуем в строку и очищаем
    value = str(value).upper().replace(',', '').strip()
    value = re.sub(r'[^\d\.\-KMBkmb]', '', value) # Убираем символы валют и лишние символы
    if value == '' or value == '-':
        return 0.0
    try:
        if 'K' in value or 'k' in value:
            return float(value.replace('K', '').replace('k', '')) * 1_000
        elif 'M' in value or 'm' in value:
            return float(value.replace('M', '').replace('m', '')) * 1_000_000
        elif 'B' in value or 'b' in value:
            return float(value.replace('B', '').replace('b', '')) * 1_000_000_000
        else:
            return float(value)
    except Exception as e:
        logger.warning(f"Ошибка парсинга строки '{value}': {e}")
        return 0.0

def parse_time_to_minutes(value):
    """Улучшенная версия парсинга времени"""
     # Преобразование массивов/серий в скаляр для надежности
    if isinstance(value, (list, np.ndarray, pd.Series)):
        if len(value) == 0:
            value = '' 
        else:
            value = value[0] if not isinstance(value, pd.Series) else (value.iloc[0] if len(value) > 0 else '')
    if pd.isna(value) or value == '' or value == 'N/A' or value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    total_minutes = 0.0
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
        # Если ничего не найдено, пробуем интерпретировать как число минут
        if total_minutes == 0:
            clean_value = re.sub(r'[^\d\.]', '', value)
            if clean_value:
                total_minutes = float(clean_value)
        return total_minutes
    except Exception as e:
         logger.warning(f"Ошибка парсинга времени '{value}': {e}")
         return 0.0

def parse_top10_holdings(value, total_top10_percent=None):
    """Улучшенная версия обработки концентрации китов с дополнительными метриками"""
    # --- Исправление: Преобразование value в скаляр ---
    if isinstance(value, (list, np.ndarray, pd.Series)):
        if len(value) == 0:
             value_scalar = '' 
        else:
             value_scalar = value[0] if not isinstance(value, pd.Series) else (value.iloc[0] if len(value) > 0 else '')
    elif pd.isna(value) or value is None: 
         value_scalar = '' 
    else:
         value_scalar = str(value) 

    # Проверка скалярного значения
    if value_scalar == '' or value_scalar == 'N/A' or value_scalar.lower() == 'nan':
        return {
            'top1_real_percent': 0.0, 'top3_real_percent': 0.0, 'top5_real_percent': 0.0,
            'concentration_ratio': 0.0, 'internal_distribution': [0.0]*10,
            'gini_coefficient': 0.0, 'herfindahl_index': 0.0
        }
    # --- Конец исправления ---
    
    try:
        # Обработка как строки
        value_clean = value_scalar.strip('[]').replace(' ', '') 
        if value_clean:
            # Фильтруем пустые и 'nan' значения
            internal_percentages = [float(x) for x in value_clean.split(',') if x.strip() and x.strip().lower() != 'nan']
        else:
             internal_percentages = []
        
        # Дополняем до 10 элементов нулями
        while len(internal_percentages) < 10:
            internal_percentages.append(0.0)
            
        # Расчет реальных процентов
        if total_top10_percent is None or pd.isna(total_top10_percent) or total_top10_percent <= 0:
            real_percentages = internal_percentages
        else:
            total_internal = sum(internal_percentages)
            if total_internal > 0:
                normalized_percentages = [x / total_internal for x in internal_percentages]
                real_percentages = [x * total_top10_percent / 100 for x in normalized_percentages]
            else:
                real_percentages = [0.0] * 10

        # Основные метрики
        top1_real = real_percentages[0] if len(real_percentages) > 0 else 0.0
        top3_real = sum(real_percentages[:3]) if len(real_percentages) >= 3 else sum(real_percentages)
        top5_real = sum(real_percentages[:5]) if len(real_percentages) >= 5 else sum(real_percentages)

        # Коэффициент концентрации
        total_internal_nonzero = sum([x for x in internal_percentages if x > 0])
        concentration_ratio = internal_percentages[0] / total_internal_nonzero if total_internal_nonzero > 0 else 0.0

        # Коэффициент Джини (неравенство распределения)
        sorted_percentages = sorted([x for x in internal_percentages if x > 0], reverse=True)
        n = len(sorted_percentages)
        if n > 1:
            cumsum = np.cumsum(sorted_percentages)
            total_sorted_sum = sum(sorted_percentages)
            if total_sorted_sum > 0:
                gini_coefficient = (n + 1 - 2 * sum((n + 1 - i) * x for i, x in enumerate(cumsum))) / (n * total_sorted_sum)
            else:
                gini_coefficient = 0.0
        else:
            gini_coefficient = 0.0

        # Индекс Херфиндаля (концентрация рынка)
        total_sum = sum(internal_percentages)
        if total_sum > 0:
            herfindahl_index = sum((x / total_sum) ** 2 for x in internal_percentages if x > 0)
        else:
            herfindahl_index = 0.0

        return {
            'top1_real_percent': float(top1_real),
            'top3_real_percent': float(top3_real),
            'top5_real_percent': float(top5_real),
            'concentration_ratio': float(concentration_ratio),
            'internal_distribution': [float(x) for x in internal_percentages],
            'gini_coefficient': float(gini_coefficient),
            'herfindahl_index': float(herfindahl_index)
        }
    except Exception as e:
        logger.error(f"Ошибка в parse_top10_holdings для значения '{value}' (scalar: '{value_scalar}'), total_top10_percent='{total_top10_percent}': {e}")
        return {
            'top1_real_percent': 0.0, 'top3_real_percent': 0.0, 'top5_real_percent': 0.0,
            'concentration_ratio': 0.0, 'internal_distribution': [0.0]*10,
            'gini_coefficient': 0.0, 'herfindahl_index': 0.0
        }

# --- Функция предсказания (адаптирована из ячейки 18 ноутбука) ---
def predict_memtoken_advanced(token_data):
    """
    Улучшенная функция предсказания успеха мемтокена
    Args:
        token_data (dict): Данные токена
    Returns:
        dict: Результат предсказания
    """
    global model, scaler, encoders, features, model_name, ensemble_weights

    if model is None or scaler is None or encoders is None or features is None:
        logger.error("Модель не загружена.")
        return {
            "success": False,
            "error": "Модель не загружена. Проверьте логи сервера.",
            "probability": 0.5,
            "prediction": 0,
            "recommendation": "❓ ОШИБКА - Модель не загружена",
            "confidence_interval": [0.0, 1.0]
        }

    try:
        logger.debug(f"Начинаем обработку данных токена: {token_data}")
        # --- Подготовка DataFrame ---
        token_df = pd.DataFrame([token_data])
        
        # --- Обработка данных (как в ноутбуке) ---
        string_cols = ['market_cap', 'liquidity', 'ath']
        for col in string_cols:
            if col in token_df.columns:
                logger.debug(f"Обработка столбца '{col}'")
                token_df[col] = token_df[col].apply(parse_string_number)
                # В API мы просто копируем обработанное значение
                token_df[f'{col}_capped'] = token_df[col] 

        if 'token_age' in token_df.columns:
            logger.debug("Обработка столбца 'token_age'")
            token_df['token_age_minutes'] = token_df['token_age'].apply(parse_time_to_minutes)
            token_df['token_age_hours'] = token_df['token_age_minutes'] / 60
            token_df['token_age_days'] = token_df['token_age_minutes'] / 1440
            token_df['is_very_new'] = (token_df['token_age_minutes'] < 60).astype(int)
            token_df['is_new'] = (token_df['token_age_minutes'] < 1440).astype(int)
            token_df['is_mature'] = (token_df['token_age_minutes'] > 10080).astype(int)
            token_df['is_very_mature'] = (token_df['token_age_minutes'] > 43200).astype(int)  # > 30 дней
            token_df['token_age_log'] = np.log1p(token_df['token_age_minutes'])
            token_df['token_age_sqrt'] = np.sqrt(token_df['token_age_minutes'])

        if 'top_10_holdings' in token_df.columns:
             logger.debug("Обработка столбца 'top_10_holdings'")
             # Получаем данные для parse_top10_holdings
             top10_total = token_data.get('top_10_percent', None) 
             holdings_str = token_data.get('top_10_holdings', '') 
             metrics = parse_top10_holdings(holdings_str, top10_total)
             token_df['biggest_whale_percent'] = metrics['top1_real_percent']
             token_df['top3_whales_percent'] = metrics['top3_real_percent']
             token_df['top5_whales_percent'] = metrics['top5_real_percent']
             token_df['whale_dominance_index'] = metrics['concentration_ratio']
             token_df['gini_coefficient'] = metrics['gini_coefficient']
             token_df['herfindahl_index'] = metrics['herfindahl_index']
             for i in range(10):
                 token_df[f'whale_{i+1}_internal_share'] = metrics['internal_distribution'][i]

        # --- Создание признаков (как в ноутбуке) ---
        logger.debug("Создание признаков...")
        # Торговые паттерны
        token_df['buy_sell_ratio_1m'] = np.where(token_df['sell_volume_1m'] > 0, 
                                               token_df['buy_volume_1m'] / token_df['sell_volume_1m'], 0)
        token_df['buy_sell_ratio_5m'] = np.where(token_df['sell_volume_5m'] > 0, 
                                               token_df['buy_volume_5m'] / token_df['sell_volume_5m'], 0)
        token_df['total_volume_1m'] = token_df['buy_volume_1m'] + token_df['sell_volume_1m']
        token_df['total_volume_5m'] = token_df['buy_volume_5m'] + token_df['sell_volume_5m']
        token_df['buy_pressure_1m'] = np.where(token_df['total_volume_1m'] > 0, 
                                             token_df['buy_volume_1m'] / token_df['total_volume_1m'], 0)
        token_df['buy_pressure_5m'] = np.where(token_df['total_volume_5m'] > 0, 
                                             token_df['buy_volume_5m'] / token_df['total_volume_5m'], 0)
        token_df['buy_pressure_change'] = token_df['buy_pressure_5m'] - token_df['buy_pressure_1m']
        token_df['avg_buy_size_1m'] = np.where(token_df['buys_1m'] > 0, 
                                   token_df['buy_volume_1m'] / token_df['buys_1m'], 0)
        token_df['avg_sell_size_1m'] = np.where(token_df['sells_1m'] > 0, 
                                    token_df['sell_volume_1m'] / token_df['sells_1m'], 0)
        token_df['avg_buy_size_5m'] = np.where(token_df['buys_5m'] > 0, 
                                   token_df['buy_volume_5m'] / token_df['buys_5m'], 0)
        token_df['avg_sell_size_5m'] = np.where(token_df['sells_5m'] > 0, 
                                    token_df['sell_volume_5m'] / token_df['sells_5m'], 0)
        token_df['buy_vs_sell_size_1m'] = np.where(token_df['avg_sell_size_1m'] > 0,
                                       token_df['avg_buy_size_1m'] / token_df['avg_sell_size_1m'], 0)
        token_df['buy_vs_sell_size_5m'] = np.where(token_df['avg_sell_size_5m'] > 0,
                                       token_df['avg_buy_size_5m'] / token_df['avg_sell_size_5m'], 0)
        token_df['volume_growth_1m_to_5m'] = np.where(token_df['volume_1m'] > 0,
                                          token_df['volume_5m'] / token_df['volume_1m'], 0)
        token_df['buy_growth_1m_to_5m'] = np.where(token_df['buys_1m'] > 0,
                                       token_df['buys_5m'] / token_df['buys_1m'], 0)
        token_df['sell_growth_1m_to_5m'] = np.where(token_df['sells_1m'] > 0,
                                        token_df['sells_5m'] / token_df['sells_1m'], 0)

        # Поведенческие паттерны (предполагаем, что данные приходят в "плоском" виде)
        logger.debug("Создание поведенческих признаков...")
        token_df['total_holders_emoji'] = (token_df.get('buyers_green', 0) + token_df.get('buyers_blue', 0) + 
                                         token_df.get('buyers_yellow', 0) + token_df.get('buyers_red', 0))
        token_df['total_snipers'] = (token_df.get('buyers_clown', 0) + token_df.get('buyers_sun', 0) + 
                                   token_df.get('buyers_moon_half', 0) + token_df.get('buyers_moon_new', 0))
        token_df['holders_keep_ratio'] = np.where(token_df['total_holders_emoji'] > 0,
                                      (token_df.get('buyers_green', 0) + token_df.get('buyers_blue', 0)) / token_df['total_holders_emoji'], 0)
        token_df['holders_sell_ratio'] = np.where(token_df['total_holders_emoji'] > 0,
                                      (token_df.get('buyers_yellow', 0) + token_df.get('buyers_red', 0)) / token_df['total_holders_emoji'], 0)
        token_df['holders_diamond_hands'] = np.where(token_df['total_holders_emoji'] > 0,
                                         token_df.get('buyers_green', 0) / token_df['total_holders_emoji'], 0)
        token_df['holders_paper_hands'] = np.where(token_df['total_holders_emoji'] > 0,
                                       token_df.get('buyers_red', 0) / token_df['total_holders_emoji'], 0)
        token_df['snipers_keep_ratio'] = np.where(token_df['total_snipers'] > 0,
                                      (token_df.get('buyers_clown', 0) + token_df.get('buyers_sun', 0)) / token_df['total_snipers'], 0)
        token_df['snipers_dump_ratio'] = np.where(token_df['total_snipers'] > 0,
                                      (token_df.get('buyers_moon_half', 0) + token_df.get('buyers_moon_new', 0)) / token_df['total_snipers'], 0)
        token_df['snipers_vs_holders_ratio'] = np.where(token_df['total_holders_emoji'] > 0,
                                            token_df['total_snipers'] / token_df['total_holders_emoji'], 0)
        token_df['total_active_addresses'] = token_df['total_holders_emoji'] + token_df['total_snipers']
        token_df['trust_score'] = (token_df.get('buyers_green', 0) + token_df.get('buyers_clown', 0)) / (token_df['total_active_addresses'] + 1)
        token_df['distrust_score'] = (token_df.get('buyers_red', 0) + token_df.get('buyers_moon_new', 0)) / (token_df['total_active_addresses'] + 1)

        # Рыночные коэффициенты
        logger.debug("Создание рыночных коэффициентов...")
        token_df['liquidity_to_mcap_ratio'] = np.where(token_df.get('market_cap_capped', 0) > 0,
                                                     token_df.get('liquidity_capped', 0) / token_df['market_cap_capped'], 0)
        token_df['volume_to_liquidity_ratio'] = np.where(token_df.get('liquidity_capped', 0) > 0,
                                                       token_df['total_volume_5m'] / token_df['liquidity_capped'], 0)
        token_df['volume_to_mcap_ratio'] = np.where(token_df.get('market_cap_capped', 0) > 0,
                                        token_df['total_volume_5m'] / token_df['market_cap_capped'], 0)
        
        token_df['volume_per_sol'] = np.where(token_df.get('sol_pooled', 0) > 0,
                                      token_df['total_volume_5m'] / token_df['sol_pooled'], 0)
        token_df['mcap_per_sol'] = np.where(token_df.get('sol_pooled', 0) > 0,
                                    token_df.get('market_cap_capped', 0) / token_df['sol_pooled'], 0)
        
        token_df['ratio_change'] = token_df.get('current_ratio', 0) - token_df.get('initial_ratio', 0)
        token_df['ratio_change_percent'] = np.where(token_df.get('initial_ratio', 1) > 0,
                                        (token_df.get('current_ratio', 0) - token_df.get('initial_ratio', 0)) / token_df['initial_ratio'], 0)

        # Концентрация и держатели
        logger.debug("Создание признаков концентрации...")
        if 'total_holders' in token_df.columns:
            token_df['volume_per_holder'] = np.where(token_df['total_holders'] > 0,
                                         token_df['total_volume_5m'] / token_df['total_holders'], 0)
            token_df['mcap_per_holder'] = np.where(token_df['total_holders'] > 0,
                                       token_df.get('market_cap_capped', 0) / token_df['total_holders'], 0)
            token_df['active_to_total_holders_ratio'] = np.where(token_df['total_holders'] > 0,
                                                     token_df['total_active_addresses'] / token_df['total_holders'], 0)
        
        if 'freshies_1d_percent' in token_df.columns and 'freshies_7d_percent' in token_df.columns:
            token_df['freshies_growth'] = token_df['freshies_7d_percent'] - token_df['freshies_1d_percent']
            token_df['veteran_ratio'] = 100 - token_df['freshies_7d_percent']
        
        if 'dev_current_balance_percent' in token_df.columns:
            token_df['dev_risk_high'] = (token_df['dev_current_balance_percent'] > 10).astype(int)
            token_df['dev_risk_medium'] = ((token_df['dev_current_balance_percent'] > 5) & 
                               (token_df['dev_current_balance_percent'] <= 10)).astype(int)

        # Безопасность
        logger.debug("Создание признаков безопасности...")
        security_features = ['security_no_mint', 'security_blacklist', 'security_burnt', 
                            'security_dev_sold', 'security_dex_paid']
        # Убедимся, что колонки существуют
        for sf in security_features:
             if sf not in token_df.columns:
                  token_df[sf] = 0 # или False для булевых
        
        token_df['security_score'] = token_df[security_features].sum(axis=1)
        token_df['security_perfect'] = (token_df['security_score'] == 5).astype(int)
        token_df['security_risky'] = (token_df['security_score'] <= 2).astype(int)

        # Логарифмические преобразования
        logger.debug("Создание логарифмических преобразований...")
        log_features = ['market_cap_capped', 'liquidity_capped', 'ath_capped', 'total_holders', 'total_volume_5m', 'sol_pooled']
        for col in log_features:
            if col in token_df.columns:
                token_df[f'{col}_log'] = np.log1p(token_df[col].fillna(0))
                token_df[f'{col}_sqrt'] = np.sqrt(token_df[col].fillna(0))
                token_df[f'{col}_inv'] = 1 / (token_df[col].fillna(0) + 1)

        # Взаимодействия признаков
        logger.debug("Создание взаимодействий признаков...")
        token_df['volume_liquidity_interaction'] = token_df['total_volume_5m'] * token_df.get('liquidity_capped_log', 0)
        token_df['volume_mcap_interaction'] = token_df['total_volume_5m'] * token_df.get('market_cap_capped_log', 0)
        token_df['age_volume_interaction'] = token_df.get('token_age_log', 0) * token_df['total_volume_5m']
        token_df['age_holders_interaction'] = token_df.get('token_age_log', 0) * np.log1p(token_df.get('total_holders', 1))
        # Используем имена признаков из ноутбука, если они отсутствуют, подставляем 0
        token_df['trust_whale_interaction'] = token_df.get('trust_score', 0) * token_df.get('whale_centralization', 0) 
        token_df['distrust_whale_interaction'] = token_df.get('distrust_score', 0) * token_df.get('dangerous_whale_concentration', 0) 

        # --- Подготовка к предсказанию ---
        logger.debug("Подготовка данных для предсказания...")
        available_features = [col for col in features if col in token_df.columns]
        missing_features = [col for col in features if col not in token_df.columns]
        
        if missing_features:
            logger.warning(f"Отсутствуют признаки: {missing_features}. Заполняем нулями.")
            for col in missing_features:
                token_df[col] = 0
        
        X_token = token_df[features].fillna(0)
        logger.debug(f"Форма подготовленных данных: {X_token.shape}")

        # Кодирование категориальных признаков (как в ноутбуке)
        logger.debug("Кодирование категориальных признаков...")
        for col in X_token.columns:
            if col in encoders:
                try:
                    le = encoders[col]
                    # Обработка неизвестных меток
                    known_vals = set(le.classes_)
                    # Используем 'unknown' если оно было в обучении, иначе первый известный класс
                    default_val = 'unknown' if 'unknown' in known_vals else (le.classes_[0] if len(le.classes_) > 0 else '')
                    # Применяем преобразование с обработкой неизвестных значений
                    X_token[col] = X_token[col].apply(lambda x: x if str(x) in known_vals else default_val)
                    X_token[col] = le.transform(X_token[col].astype(str))
                    logger.debug(f"Закодирован признак '{col}'")
                except Exception as e:
                     logger.error(f"Ошибка кодирования '{col}': {e}. Заполняем 0.")
                     X_token[col] = 0 # Заполнение при ошибке

        # Масштабирование (как в ноутбуке)
        logger.debug("Масштабирование данных...")
        X_token_scaled = scaler.transform(X_token)

        # Предсказание (как в ноутбуке)
        logger.debug("Выполнение предсказания...")
        if model_name == 'Ensemble' and isinstance(model, dict) and ensemble_weights is not None:
            logger.debug("Используется ансамбль.")
            ensemble_probas = []
            model_names = list(model.keys()) 
            for name in model_names:
                m = model[name]
                try:
                    prob = m.predict_proba(X_token_scaled)[0, 1]
                    ensemble_probas.append(prob)
                    logger.debug(f"Модель {name} предсказала вероятность: {prob}")
                except Exception as e:
                    logger.error(f"Ошибка предсказания модели {name}: {e}")
                    ensemble_probas.append(0.5) 
            
            if ensemble_probas:
                 probability = np.average(ensemble_probas, weights=ensemble_weights)
                 confidence_interval = (np.min(ensemble_probas), np.max(ensemble_probas))
                 logger.debug(f"Ансамбль: средняя вероятность={probability}, интервал={confidence_interval}")
            else:
                 probability = 0.5
                 confidence_interval = (0.0, 1.0)
        else:
            logger.debug(f"Используется одиночная модель: {model_name}.")
            try:
                probability = model.predict_proba(X_token_scaled)[0, 1]
                # Простая оценка доверительного интервала
                confidence_interval = (max(probability * 0.9, 0.0), min(probability * 1.1, 1.0))
                logger.debug(f"Одиночная модель: вероятность={probability}")
            except Exception as e:
                 logger.error(f"Ошибка предсказания основной модели: {e}")
                 probability = 0.5
                 confidence_interval = (0.0, 1.0)

        prediction = int(probability > 0.5)
        
        if probability >= 0.85:
            recommendation = "🔥 VERY STRONG BUY - Исключительно высокие шансы на рост!"
        elif probability >= 0.75:
            recommendation = "🚀 STRONG BUY - Очень высокие шансы на рост"
        elif probability >= 0.65:
            recommendation = "✅ BUY - Хорошие шансы на рост"
        elif probability >= 0.55:
            recommendation = "⚖️ CONSIDER - Умеренные шансы, анализируйте дополнительно"
        elif probability >= 0.45:
            recommendation = "⚠️ CAUTION - Низкие шансы, высокий риск"
        elif probability >= 0.35:
            recommendation = "❌ AVOID - Очень низкие шансы на рост"
        else:
            recommendation = "🚫 STRONG AVOID - Крайне низкие шансы"

        # Добавляем информацию о токене для удобства
        result = {
            "success": True,
            "probability": float(probability),
            "prediction": prediction, 
            "recommendation": recommendation,
            "confidence_interval": [float(confidence_interval[0]), float(confidence_interval[1])]
        }
        # Добавляем символ и имя токена, если они есть
        if 'symbol' in token_data:
            result['token_symbol'] = token_data['symbol']
        if 'name' in token_data:
            result['token_name'] = token_data['name']

        return result

    except Exception as e:
        logger.error(f"Ошибка в predict_memtoken_advanced: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": f"Ошибка обработки данных или предсказания: {str(e)}",
            "probability": 0.5,
            "prediction": 0,
            "recommendation": "❓ ОШИБКА - Не удалось обработать данные",
            "confidence_interval": [0.0, 1.0]
        }

# --- Загрузка модели при запуске (адаптирована из ячейки 20 ноутбука) ---
def load_model():
    """Загружает обученную модель и её компоненты."""
    global model, scaler, encoders, features, model_name, ensemble_weights
    try:
        # Имена файлов модели (должны совпадать с именами файлов, которые вы загружаете на Railway)
        model_path = 'memtoken_model_improved.pkl'
        scaler_path = 'memtoken_scaler_improved.pkl'
        encoders_path = 'memtoken_encoders_improved.pkl' 
        features_path = 'memtoken_features_improved.pkl'
        metadata_path = 'memtoken_model_metadata.json'
        ensemble_weights_path = 'memtoken_ensemble_weights.pkl'

        logger.info("Начинаем загрузку модели...")
        
        # Проверяем существование файлов
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Файл модели {model_path} не найден.")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Файл скейлера {scaler_path} не найден.")
        if not os.path.exists(encoders_path):
            raise FileNotFoundError(f"Файл энкодеров {encoders_path} не найден.")
        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Файл признаков {features_path} не найден.")

        logger.info(f"Файл модели найден. Размер: {os.path.getsize(model_path)} байт")
        logger.info(f"Файл скейлера найден. Размер: {os.path.getsize(scaler_path)} байт")
        logger.info(f"Файл энкодеров найден. Размер: {os.path.getsize(encoders_path)} байт")
        logger.info(f"Файл признаков найден. Размер: {os.path.getsize(features_path)} байт")

        # Загружаем компоненты
        model = joblib.load(model_path)
        logger.info("Модель загружена.")
        
        scaler = joblib.load(scaler_path)
        logger.info("Скейлер загружен.")
        
        encoders = joblib.load(encoders_path) 
        logger.info("Энкодеры загружены.")
        
        features = joblib.load(features_path) 
        logger.info(f"Признаки загружены. Количество: {len(features) if features else 'N/A'}")
        
        # Определяем тип модели из метаданных
        if os.path.exists(metadata_path):
             with open(metadata_path, 'r') as f:
                 metadata = json.load(f)
             model_name = metadata.get('best_model_name', 'Unknown')
             logger.info(f"Метаданные загружены. Тип модели: {model_name}")
             if model_name == 'Ensemble' and os.path.exists(ensemble_weights_path):
                  ensemble_weights = np.array(joblib.load(ensemble_weights_path))
                  logger.info("Веса ансамбля загружены.")
             elif model_name == 'Ensemble':
                  logger.warning(f"Модель - ансамбль, но файл весов {ensemble_weights_path} не найден.")
        else:
             # Предположим, если это словарь, то ансамбль
             model_name = 'Ensemble' if isinstance(model, dict) else 'Single_Model'
             logger.info(f"Предположительно загружена модель: {model_name}")

        logger.info("✅ Все компоненты модели успешно загружены.")

    except Exception as e:
        logger.error(f"❌ КРИТИЧЕСКАЯ ОШИБКА загрузки модели: {e}", exc_info=True)
        # На Railway это приведет к тому, что приложение не сможет обрабатывать запросы,
        # но логи будут содержать сообщение об ошибке.

# --- Маршруты Flask ---

@app.route('/')
def home():
    return jsonify({
        "message": "API для предсказания успеха мемтокенов",
        "status": "running",
        "timestamp": datetime.utcnow().isoformat() + "Z"
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Endpoint для получения предсказания."""
    try:
        # Получаем сырые JSON данные из запроса
        raw_data = request.get_json()
        logger.debug(f"Получен сырые JSON: {raw_data}")

        # --- Адаптация для структуры входящих данных с "body" ---
        # Проверка и извлечение данных из "body", если они там
        if isinstance(raw_data, dict) and 'body' in raw_data:
            try:
                # Предполагаем, что 'body' - это JSON строка
                data = json.loads(raw_data['body'])
                logger.debug(f"Извлечены данные из 'body': {data}")
            except json.JSONDecodeError as e:
                logger.error(f"Ошибка декодирования JSON из 'body': {e}")
                return jsonify({"error": "Неверный формат JSON в поле 'body'"}), 400
        else:
            # Предполагаем, что данные пришли напрямую
            data = raw_data
        # --- Конец адаптации ---

        if not data:
            return jsonify({"error": "JSON данные не предоставлены"}), 400

        # Обработка входных данных (список или словарь)
        if isinstance(data, list) and len(data) > 0:
             token_data = data[0] # Берем первый токен из списка
        elif isinstance(data, dict):
             token_data = data
        else:
             return jsonify({"error": "Неверный формат данных. Ожидается JSON объект или массив с объектом."}), 400

        # --- Форматирование входных данных под ожидаемый формат функции ---
        # Преобразуем вложенные объекты в плоские поля
        formatted_data = {}

        # Базовые поля
        for key, value in token_data.items():
            if key not in ['first_buyers', 'current_initial_ratio', 'security']:
                formatted_data[key] = value

        # first_buyers
        if 'first_buyers' in token_data and isinstance(token_data['first_buyers'], dict):
            fb = token_data['first_buyers']
            formatted_data['buyers_green'] = fb.get('green', 0)
            formatted_data['buyers_blue'] = fb.get('blue', 0)
            formatted_data['buyers_yellow'] = fb.get('yellow', 0)
            formatted_data['buyers_red'] = fb.get('red', 0)
            formatted_data['buyers_clown'] = fb.get('clown', 0)
            formatted_data['buyers_sun'] = fb.get('sun', 0)
            formatted_data['buyers_moon_half'] = fb.get('moon_half', 0)
            formatted_data['buyers_moon_new'] = fb.get('moon_new', 0)

        # current_initial_ratio
        if 'current_initial_ratio' in token_data and isinstance(token_data['current_initial_ratio'], dict):
            cir = token_data['current_initial_ratio']
            formatted_data['current_ratio'] = cir.get('current', 0)
            formatted_data['initial_ratio'] = cir.get('initial', 0)

        # security
        if 'security' in token_data and isinstance(token_data['security'], dict):
            sec = token_data['security']
            formatted_data['security_no_mint'] = int(sec.get('no_mint', False))
            formatted_data['security_blacklist'] = int(sec.get('blacklist', False))
            formatted_data['security_burnt'] = int(sec.get('burnt', False))
            formatted_data['security_dev_sold'] = int(sec.get('dev_sold', False))
            formatted_data['security_dex_paid'] = int(sec.get('dex_paid', False))

        # --- Вызов функции предсказания ---
        result = predict_memtoken_advanced(formatted_data)

        return jsonify(result)

    except Exception as e:
        logger.error(f"Ошибка в /predict: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Внутренняя ошибка сервера: {str(e)}"
        }), 500

# --- Запуск приложения ---
if __name__ == '__main__':
    # Загружаем модель один раз при старте
    load_model()
    
    # Получаем порт из переменной окружения (Railway) или используем 5000 по умолчанию
    port = int(os.environ.get('PORT', 5000))
    
    # Запуск Flask приложения (на Railway будет использоваться Gunicorn)
    # host='0.0.0.0' необходимо для Railway
    app.run(host='0.0.0.0', port=port, debug=False) 
