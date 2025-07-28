# 🚀 Memtoken Prediction API

API для предсказания успеха мемтокенов на основе машинного обучения.

## 🌟 Возможности

- **Предсказание одиночных токенов** - анализ и прогноз для отдельного токена
- **Пакетное предсказание** - обработка до 100 токенов за раз
- **Детальный анализ** - углубленный анализ с факторами риска
- **Извлечение признаков** - получение всех вычисленных характеристик токена

## 🛠️ Технологии

- **Backend**: Flask, Python 3.11
- **ML**: pandas, numpy (готово для интеграции с scikit-learn, lightgbm)
- **Deploy**: Railway.com
- **CORS**: Поддержка кросс-доменных запросов

## 📡 API Endpoints

### `GET /`
Проверка состояния API

**Response:**
```json
{
  "status": "healthy",
  "message": "Memtoken Prediction API is running",
  "timestamp": "2025-01-XX...",
  "version": "1.0.0"
}
```

### `POST /predict/batch`
Пакетное предсказание для нескольких токенов (до 100)

**Request Body:**
```json
[
  {
    "symbol": "TOKEN1",
    "market_cap": "50K",
    // ... остальные поля как в /predict
  },
  {
    "symbol": "TOKEN2", 
    "market_cap": "100K",
    // ... остальные поля
  }
]
```

**Response:**
```json
{
  "results": [
    {
      "probability": 0.654,
      "prediction": 1,
      "recommendation": "✅ BUY",
      "confidence_interval": [0.554, 0.754],
      "token_info": {
        "symbol": "TOKEN1",
        "batch_index": 0
      }
    }
  ],
  "batch_stats": {
    "total_tokens": 2,
    "successful_predictions": 2,
    "failed_predictions": 0,
    "average_probability": 0.627,
    "timestamp": "2025-01-XX..."
  }
}
```

### `POST /analyze`
Детальный анализ токена с факторами риска

**Request Body:** Тот же формат что и `/predict`

**Response:**
```json
{
  "token_info": {
    "symbol": "BABU",
    "name": "First dog with residence"
  },
  "prediction": {
    "probability": 0.654,
    "recommendation": "✅ BUY"
  },
  "detailed_analysis": {
    "market_metrics": {
      "market_cap": "95.1K",
      "liquidity": "36.2K",
      "liquidity_to_mcap_ratio": 0.381,
      "assessment": "Good"
    },
    "age_analysis": {
      "token_age": "7m",
      "age_minutes": 7,
      "category": "Very New",
      "risk_level": "High"
    },
    "trading_activity": {
      "volume_5m": 225140.15,
      "buy_pressure_5m": 0.524,
      "activity_level": "High"
    },
    "whale_concentration": {
      "biggest_whale_percent": 4.2,
      "top3_whales_percent": 10.9,
      "risk_assessment": "Low Risk"
    },
    "holder_behavior": {
      "trust_score": 0.043,
      "distrust_score": 0.732,
      "sentiment": "Negative"
    },
    "security_analysis": {
      "security_score": 4,
      "overall_security": "High"
    }
  },
  "risk_factors": [
    "Very new token (<1 hour)",
    "Low buying pressure"
  ],
  "positive_factors": [
    "High security standards",
    "Low whale concentration"
  ]
}
```

### `POST /features`
Извлечение всех признаков из данных токена

**Request Body:** Тот же формат что и `/predict`

**Response:**
```json
{
  "categorized_features": {
    "basic_info": {
      "symbol": "BABU",
      "token_age_minutes": 7
    },
    "market_data": {
      "market_cap_capped": 95100,
      "liquidity_capped": 36200,
      "liquidity_to_mcap_ratio": 0.381
    },
    "trading_metrics": {
      "buy_pressure_5m": 0.524,
      "buy_sell_ratio_5m": 1.102
    },
    "holder_behavior": {
      "trust_score": 0.043,
      "holders_diamond_hands": 0.043
    },
    "whale_analysis": {
      "biggest_whale_percent": 4.22,
      "gini_coefficient": 0.89
    },
    "security": {
      "security_score": 4,
      "security_no_mint": true
    }
  },
  "all_features": {
    // ... все вычисленные признаки
  },
  "feature_count": 67,
  "timestamp": "2025-01-XX..."
}
```

## 🎯 Интерпретация результатов

### Уровни рекомендаций:
- 🔥 **VERY STRONG BUY** (0.85+) - Исключительно высокие шансы
- 🚀 **STRONG BUY** (0.75-0.85) - Очень высокие шансы  
- ✅ **BUY** (0.65-0.75) - Хорошие шансы
- ⚖️ **CONSIDER** (0.55-0.65) - Умеренные шансы
- ⚠️ **CAUTION** (0.45-0.55) - Низкие шансы
- ❌ **AVOID** (0.35-0.45) - Очень низкие шансы
- 🚫 **STRONG AVOID** (<0.35) - Крайне низкие шансы

### Ключевые факторы:
- **Возраст токена**: Очень новые (<1ч) и очень старые (>30д) токены рискованнее
- **Давление покупок**: >0.6 хорошо, <0.4 плохо
- **Концентрация китов**: >20% опасно, <10% безопасно
- **Trust Score**: >0.6 хорошо, <0.3 плохо
- **Безопасность**: 4-5 из 5 критериев желательно

## 🚀 Деплой на Railway

### 1. Подготовка проекта
```bash
# Клонируйте репозиторий или создайте новую папку
mkdir memtoken-api
cd memtoken-api

# Скопируйте все файлы из артефактов:
# - app.py
# - requirements.txt
# - Procfile
# - runtime.txt
# - railway.json
```

### 2. Деплой
1. Зайдите на [railway.app](https://railway.app)
2. Подключите GitHub репозиторий или загрузите файлы
3. Railway автоматически определит Python проект и развернет его
4. После деплоя вы получите URL типа `https://your-app.railway.app`

### 3. Настройка переменных окружения (опционально)
В Railway Dashboard можно добавить:
- `DEBUG=false` (для продакшена)
- `PORT=5000` (если нужно)

## 🧪 Тестирование

### Локальное тестирование
```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск локально
python app.py

# Тестирование API
python test_api.py
```

### Тестирование на Railway
```bash
# Обновите URL в test_api.py
API_BASE_URL = "https://your-app.railway.app"

# Запустите тесты
python test_api.py
```

## 📊 Примеры использования

### Python
```python
import requests

url = "https://your-app.railway.app/predict"
data = {
    "symbol": "DOGE",
    "market_cap": "50K",
    "liquidity": "20K",
    "token_age": "2h",
    # ... остальные поля
}

response = requests.post(url, json=data)
result = response.json()

print(f"Рекомендация: {result['recommendation']}")
print(f"Вероятность: {result['probability']:.3f}")
```

### JavaScript
```javascript
const url = 'https://your-app.railway.app/predict';
const data = {
    symbol: 'DOGE',
    market_cap: '50K',
    liquidity: '20K',
    token_age: '2h',
    // ... остальные поля
};

fetch(url, {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify(data),
})
.then(response => response.json())
.then(result => {
    console.log('Рекомендация:', result.recommendation);
    console.log('Вероятность:', result.probability);
});
```

### cURL
```bash
curl -X POST https://your-app.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "DOGE",
    "market_cap": "50K",
    "liquidity": "20K",
    "token_age": "2h",
    "volume_5m": 10000,
    "buy_volume_5m": 6000,
    "sell_volume_5m": 4000,
    "buys_5m": 50,
    "sells_5m": 30,
    "total_holders": 100,
    "first_buyers": {
      "green": 10,
      "blue": 5,
      "yellow": 3,
      "red": 2,
      "clown": 1,
      "sun": 0,
      "moon_half": 0,
      "moon_new": 0
    },
    "top_10_percent": 15,
    "top_10_holdings": [5, 3, 2, 1.5, 1, 0.8, 0.6, 0.4, 0.3, 0.2],
    "security": {
      "no_mint": true,
      "blacklist": true,
      "burnt": true,
      "dev_sold": false,
      "dex_paid": true
    }
  }'
```

## ⚠️ Ограничения

- **Пакетные запросы**: Максимум 100 токенов за раз
- **Rate limiting**: Рекомендуется не более 1000 запросов в час
- **Timeout**: 30 секунд на запрос
- **Размер данных**: Максимум 1MB на запрос

## 🔧 Интеграция с ML моделью

Текущая версия использует простые правила для демонстрации. Для интеграции с обученной моделью:

1. Добавьте joblib в requirements.txt
2. Загрузите файлы модели в проект
3. Замените функцию `simple_predict()` на загрузку и использование реальной модели

```python
import joblib

# Загрузка модели при старте приложения
model = joblib.load('memtoken_model_improved.pkl')
scaler = joblib.load('memtoken_scaler_improved.pkl')
encoders = joblib.load('memtoken_encoders_improved.pkl')

def ml_predict(features):
    # Подготовка данных
    X = prepare_features(features)
    X_scaled = scaler.transform(X)
    
    # Предсказание
    probability = model.predict_proba(X_scaled)[0, 1]
    prediction = model.predict(X_scaled)[0]
    
    return probability, prediction
```

## 📞 Поддержка

При возникновении проблем:

1. Проверьте логи в Railway Dashboard
2. Убедитесь что все обязательные поля присутствуют в запросе
3. Проверьте формат данных (числа как числа, строки как строки)
4. Используйте `test_api.py` для диагностики

## 📝 Лицензия

MIT License - используйте свободно для коммерческих и некоммерческих проектов.

---

🚀 **API готов к использованию!** Развертывайте на Railway и начинайте предсказывать успех мемтокенов!
```

### `POST /predict`
Предсказание для одного токена

**Request Body:**
```json
{
  "symbol": "BABU",
  "name": "First dog with residence",
  "contract_address": "yaFH5SUG6XTY2UAXSvGXfRPphNXbLjHR6QSad7Tbonk",
  "token_age": "7m",
  "market_cap": "95.1K",
  "liquidity": "36.2K",
  "sol_pooled": 86.98,
  "ath": "101.1K",
  "volume_1m": 41985.64,
  "buy_volume_1m": 21714.81,
  "sell_volume_1m": 20270.83,
  "buys_1m": 208,
  "sells_1m": 189,
  "volume_5m": 225140.15,
  "buy_volume_5m": 118049.78,
  "sell_volume_5m": 107090.36,
  "buys_5m": 1009,
  "sells_5m": 796,
  "first_buyers": {
    "green": 3,
    "blue": 3,
    "yellow": 10,
    "red": 51,
    "clown": 0,
    "sun": 0,
    "moon_half": 1,
    "moon_new": 2
  },
  "current_initial_ratio": {
    "current": 11.14,
    "initial": 74.58
  },
  "total_holders": 420,
  "freshies_1d_percent": 10,
  "freshies_7d_percent": 23,
  "top_10_percent": 22,
  "top_10_holdings": [19.18, 3.84, 2.9, 2.27, 2.2, 2.19, 1.97, 1.7, 1.25, 1.23],
  "dev_current_balance_percent": 0.79,
  "security": {
    "no_mint": true,
    "blacklist": true,
    "burnt": true,
    "dev_sold": true,
    "dex_paid": false
  }
}
```

**Response:**
```json
{
  "probability": 0.654,
  "prediction": 1,
  "recommendation": "✅ BUY",
  "confidence_interval": [0.554, 0.754],
  "key_factors": {
    "age_minutes": 7,
    "buy_pressure_5m": 0.524,
    "biggest_whale_percent": 4.2,
    "trust_score": 0.043,
    "security_score": 4
  },
  "token_info": {
    "symbol": "BABU",
    "name": "First dog with residence",
    "contract_address": "yaFH5SUG6XTY2UAXSvGXfRPphNXbLjHR6QSad7Tbonk",
    "market_cap": "95.1K",
    "liquidity": "36.2K",
    "token_age": "7m",
    "timestamp": "2025-01-XX..."
  }
