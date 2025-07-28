import json
import os
import joblib
import numpy as np
import pandas as pd
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
from sklearn.preprocessing import RobustScaler, LabelEncoder

app = FastAPI(title="Memtoken Prediction API", description="API for predicting memtoken success", version="1.0")

# Загрузка метаданных и компонентов модели при старте приложения
try:
    with open('memtoken_model_metadata.json', 'r') as f:
        metadata = json.load(f)
    best_model_name = metadata['best_model_name']
    final_model = joblib.load('memtoken_model_improved.pkl')
    scaler: RobustScaler = joblib.load('memtoken_scaler_improved.pkl')
    label_encoders: Dict[str, LabelEncoder] = joblib.load('memtoken_encoders_improved.pkl')
    features: List[str] = joblib.load('memtoken_features_improved.pkl')
    if best_model_name == 'Ensemble':
        ensemble_weights = np.array(joblib.load('memtoken_ensemble_weights.pkl'))
except FileNotFoundError as e:
    raise RuntimeError(f"Missing model file: {e.filename}")

# Вспомогательные функции
def parse_string_number(value):
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
    if pd.isna(value) or value == '' or value == 'N/A':
        return 0
    if isinstance(value, (int, float)):
        return float(value)
    total_minutes = 0
    value = str(value).lower().strip()
    try:
        days = re.findall(r'(\d+(?:\.\d+)?)d', value)
        if days:
            total_minutes += float(days[0]) * 1440
        hours = re.findall(r'(\d+(?:\.\d+)?)h', value)
        if hours:
            total_minutes += float(hours[0]) * 60
        minutes = re.findall(r'(\d+(?:\.\d+)?)m(?!s)', value)
        if minutes:
            total_minutes += float(minutes[0])
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

def parse_top10_holdings(value, total_top10_percent=None):
    if pd.isna(value) or value == '' or value == 'N/A':
        return {
            'top1_real_percent': 0, 'top3_real_percent': 0, 'top5_real_percent': 0,
            'concentration_ratio': 0, 'internal_distribution': [0]*10,
            'gini_coefficient': 0, 'herfindahl_index': 0
        }
    try:
        value_clean = str(value).strip('[]').replace(' ', '')
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
        top1_real = real_percentages[0] if len(real_percentages) > 0 else 0
        top3_real = sum(real_percentages[:3]) if len(real_percentages) >= 3 else sum(real_percentages)
        top5_real = sum(real_percentages[:5]) if len(real_percentages) >= 5 else sum(real_percentages)
        total_internal_nonzero = sum([x for x in internal_percentages if x > 0])
        concentration_ratio = internal_percentages[0] / total_internal_nonzero if total_internal_nonzero > 0 else 0
        sorted_percentages = sorted([x for x in internal_percentages if x > 0], reverse=True)
        n = len(sorted_percentages)
        if n > 1:
            cumsum = np.cumsum(sorted_percentages)
            gini_coefficient = (n + 1 - 2 * sum((n + 1 - i) * x for i, x in enumerate(cumsum))) / (n * sum(sorted_percentages))
        else:
            gini_coefficient = 0
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
            'internal_distribution': internal_percentages,
            'gini_coefficient': gini_coefficient,
            'herfindahl_index': herfindahl_index
        }
    except Exception:
        return {
            'top1_real_percent': 0, 'top3_real_percent': 0, 'top5_real_percent': 0,
            'concentration_ratio': 0, 'internal_distribution': [0]*10,
            'gini_coefficient': 0, 'herfindahl_index': 0
        }

# Функция предсказания
def predict_memtoken_advanced(token_data: Dict[str, Any]):
    token_df = pd.DataFrame([token_data])
    
    try:
        # Обработка данных
        string_cols = ['market_cap', 'liquidity', 'ath']
        for col in string_cols:
            if col in token_df.columns:
                token_df[col] = token_df[col].apply(parse_string_number)
                token_df[f'{col}_capped'] = token_df[col]  # Упрощено, добавьте clip если нужно

        if 'token_age' in token_df.columns:
            token_df['token_age_minutes'] = token_df['token_age'].apply(parse_time_to_minutes)
            token_df['token_age_hours'] = token_df['token_age_minutes'] / 60
            token_df['token_age_days'] = token_df['token_age_minutes'] / 1440
            token_df['is_very_new'] = (token_df['token_age_minutes'] < 60).astype(int)
            token_df['is_new'] = (token_df['token_age_minutes'] < 1440).astype(int)
            token_df['is_mature'] = (token_df['token_age_minutes'] > 10080).astype(int)
            token_df['is_very_mature'] = (token_df['token_age_minutes'] > 43200).astype(int)
            token_df['token_age_log'] = np.log1p(token_df['token_age_minutes'])
            token_df['token_age_sqrt'] = np.sqrt(token_df['token_age_minutes'])

        if 'top_10_holdings' in token_df.columns and 'top_10_percent' in token_df.columns:
            holdings_metrics = [parse_top10_holdings(row['top_10_holdings'], row['top_10_percent']) for i, row in token_df.iterrows()]
            token_df['biggest_whale_percent'] = [x['top1_real_percent'] for x in holdings_metrics]
            token_df['top3_whales_percent'] = [x['top3_real_percent'] for x in holdings_metrics]
            token_df['top5_whales_percent'] = [x['top5_real_percent'] for x in holdings_metrics]
            token_df['whale_dominance_index'] = [x['concentration_ratio'] for x in holdings_metrics]
            token_df['gini_coefficient'] = [x['gini_coefficient'] for x in holdings_metrics]
            token_df['herfindahl_index'] = [x['herfindahl_index'] for x in holdings_metrics]
            for i in range(10):
                token_df[f'whale_{i+1}_internal_share'] = [x['internal_distribution'][i] for x in holdings_metrics]

        # Добавьте все остальные признаки из вашего кода здесь (торговые паттерны, поведенческие, рыночные и т.д.)
        # Пример:
        token_df['buy_sell_ratio_1m'] = np.where(token_df.get('sell_volume_1m', 0) > 0, token_df.get('buy_volume_1m', 0) / token_df.get('sell_volume_1m', 0), 0)
        # ... (скопируйте все остальные признаки)

        # Подготовка X_token
        X_token = token_df.reindex(columns=features, fill_value=0)

        # Кодирование
        for col in X_token.columns:
            if col in label_encoders:
                try:
                    X_token[col] = label_encoders[col].transform(X_token[col].astype(str))
                except ValueError:
                    X_token[col] = 0  # Если неизвестная категория

        X_token_scaled = scaler.transform(X_token)

        # Предсказание
        if best_model_name == 'Ensemble':
            ensemble_probas = []
            for name, model in final_model.items():
                prob = model.predict_proba(X_token_scaled)[0, 1]
                ensemble_probas.append(prob)
            probability = np.average(ensemble_probas, weights=ensemble_weights)
            prediction = (probability > 0.5).astype(int)
            confidence_interval = (np.min(ensemble_probas), np.max(ensemble_probas))
        else:
            probability = final_model.predict_proba(X_token_scaled)[0, 1]
            prediction = final_model.predict(X_token_scaled)[0]
            confidence_interval = (probability * 0.9, min(probability * 1.1, 1.0))

        # Рекомендация
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

        return probability, prediction, recommendation, confidence_interval

    except Exception as e:
        raise ValueError(f"Ошибка в предсказании: {str(e)}")

# Pydantic модель для данных токена
class TokenData(BaseModel):
    symbol: str
    name: str
    contract_address: str
    token_age: str
    views: int
    market_cap: str
    liquidity: str
    sol_pooled: float
    ath: str
    ath_change_percent: int
    ath_time_ago: str
    volume_1m: float
    buy_volume_1m: float
    sell_volume_1m: float
    buys_1m: int
    sells_1m: int
    volume_5m: float
    buy_volume_5m: float
    sell_volume_5m: float
    buys_5m: int
    sells_5m: int
    first_buyers: Dict[str, Any]
    current_initial_ratio: Dict[str, float]
    total_holders: int
    freshies_1d_percent: int
    freshies_7d_percent: int
    top_10_percent: int
    top_10_holdings: List[float]
    dev_current_balance_percent: float
    dev_sol_balance: float
    security: Dict[str, bool]

# Endpoint для предсказания (принимает массив токенов)
@app.post("/predict", response_model=List[Dict[str, Any]])
def predict(tokens: List[TokenData]):
    results = []
    for token in tokens:
        try:
            prob, pred, rec, conf = predict_memtoken_advanced(token.dict())
            results.append({
                "symbol": token.symbol,
                "probability": float(prob),
                "prediction": int(pred),
                "recommendation": rec,
                "confidence_interval": [float(conf[0]), float(conf[1])]
            })
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return results

# Root endpoint
@app.get("/")
def root():
    return {"message": "Memtoken Prediction API is running", "version": "1.0", "model": best_model_name}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
