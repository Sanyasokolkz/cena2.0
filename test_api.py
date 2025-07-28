#!/usr/bin/env python3
"""
Скрипт для тестирования API предсказания мемтокенов
"""

import requests
import json
import time

# Конфигурация
API_BASE_URL = "http://localhost:5000"  # Замените на URL Railway после деплоя
# API_BASE_URL = "https://your-app.railway.app"

# Тестовые данные (пример из вашего запроса)
TEST_TOKEN = {
    "symbol": "BABU",
    "name": "First dog with residence",
    "contract_address": "yaFH5SUG6XTY2UAXSvGXfRPphNXbLjHR6QSad7Tbonk",
    "token_age": "7m",
    "views": 226,
    "market_cap": "95.1K",
    "liquidity": "36.2K",
    "sol_pooled": 86.98,
    "ath": "101.1K",
    "ath_change_percent": -16,
    "ath_time_ago": "1m",
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
        "visual_map": "🌓🌚🌚🟡🟡🟡⭕⭕⭕⭕⭕🟡🟡⭕⭕⭕⭕⭕⭕⭕⭕⭕⭕⭕⭕⭕⭕🟡⭕⭕⭕⭕⭕⭕🟡⭕🟢⭕🔵⭕⭕⭕⭕🟢⭕🟡⭕⭕⭕⭕⭕⭕⭕⭕⭕⭕🔵⭕🟡🟡⭕⭕🔵🟢⭕⭕⭕⭕⭕⭕",
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
    "dev_sol_balance": 25.0321,
    "security": {
        "no_mint": True,
        "blacklist": True,
        "burnt": True,
        "dev_sold": True,
        "dex_paid": False
    }
}

def test_health_check():
    """Тест проверки здоровья API"""
    print("🏥 Тестируем health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_single_prediction():
    """Тест предсказания для одного токена"""
    print("\n🔮 Тестируем предсказание для одного токена...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json=TEST_TOKEN,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if response.status_code == 200:
            print(f"\n📊 Результат предсказания:")
            print(f"   Символ: {result['token_info']['symbol']}")
            print(f"   Вероятность: {result['probability']:.3f}")
            print(f"   Предсказание: {'Успешный' if result['prediction'] == 1 else 'Неуспешный'}")
            print(f"   Рекомендация: {result['recommendation']}")
            print(f"   Доверительный интервал: {result['confidence_interval']}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_batch_prediction():
    """Тест пакетного предсказания"""
    print("\n📦 Тестируем пакетное предсказание...")
    
    # Создаем несколько тестовых токенов
    batch_tokens = []
    for i in range(3):
        token = TEST_TOKEN.copy()
        token['symbol'] = f"TEST{i+1}"
        token['name'] = f"Test Token {i+1}"
        token['market_cap'] = f"{(i+1)*50}K"
        batch_tokens.append(token)
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict/batch",
            json=batch_tokens,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"\n📊 Статистика пакета:")
            stats = result['batch_stats']
            print(f"   Всего токенов: {stats['total_tokens']}")
            print(f"   Успешных предсказаний: {stats['successful_predictions']}")
            print(f"   Средняя вероятность: {stats['average_probability']:.3f}")
            
            print(f"\n📋 Результаты:")
            for i, res in enumerate(result['results'][:3]):  # Показываем первые 3
                if 'error' not in res:
                    print(f"   {i+1}. {res['token_info']['symbol']}: {res['recommendation']} ({res['probability']:.3f})")
                else:
                    print(f"   {i+1}. Ошибка: {res['message']}")
        else:
            print(f"Response: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_detailed_analysis():
    """Тест детального анализа"""
    print("\n🔍 Тестируем детальный анализ...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            json=TEST_TOKEN,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"\n📊 Детальный анализ:")
            print(f"   Символ: {result['token_info']['symbol']}")
            print(f"   Рекомендация: {result['prediction']['recommendation']}")
            
            analysis = result['detailed_analysis']
            print(f"\n💰 Рыночные метрики:")
            print(f"   Market Cap: {analysis['market_metrics']['market_cap']}")
            print(f"   Оценка ликвидности: {analysis['market_metrics']['assessment']}")
            
            print(f"\n⏰ Анализ возраста:")
            print(f"   Возраст: {analysis['age_analysis']['token_age']}")
            print(f"   Категория: {analysis['age_analysis']['category']}")
            print(f"   Уровень риска: {analysis['age_analysis']['risk_level']}")
            
            print(f"\n🐋 Концентрация китов:")
            print(f"   Главный кит: {analysis['whale_concentration']['biggest_whale_percent']:.1f}%")
            print(f"   Оценка риска: {analysis['whale_concentration']['risk_assessment']}")
            
            print(f"\n🔒 Безопасность:")
            print(f"   Общая безопасность: {analysis['security_analysis']['overall_security']}")
            
            if result['risk_factors']:
                print(f"\n⚠️ Факторы риска:")
                for factor in result['risk_factors']:
                    print(f"   • {factor}")
            
            if result['positive_factors']:
                print(f"\n✅ Положительные факторы:")
                for factor in result['positive_factors']:
                    print(f"   • {factor}")
        else:
            print(f"Response: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def test_feature_extraction():
    """Тест извлечения признаков"""
    print("\n🛠️  Тестируем извлечение признаков...")
    try:
        response = requests.post(
            f"{API_BASE_URL}/features",
            json=TEST_TOKEN,
            headers={'Content-Type': 'application/json'}
        )
        print(f"Status: {response.status_code}")
        result = response.json()
        
        if response.status_code == 200:
            print(f"\n📊 Извлеченные признаки:")
            print(f"   Всего признаков: {result['feature_count']}")
            
            categories = result['categorized_features']
            print(f"\n📋 Категории признаков:")
            for category, features in categories.items():
                print(f"   {category}: {len(features)} признаков")
            
            # Показываем некоторые ключевые признаки
            print(f"\n🔑 Ключевые признаки:")
            market = categories['market_data']
            print(f"   Liquidity/MCap ratio: {market['liquidity_to_mcap_ratio']:.3f}")
            
            trading = categories['trading_metrics']
            print(f"   Buy pressure (5m): {trading['buy_pressure_5m']:.3f}")
            
            whale = categories['whale_analysis']
            print(f"   Biggest whale: {whale['biggest_whale_percent']:.1f}%")
        else:
            print(f"Response: {json.dumps(result, indent=2)}")
        
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def run_performance_test():
    """Тест производительности"""
    print("\n⚡ Тестируем производительность...")
    
    num_requests = 10
    start_time = time.time()
    successful_requests = 0
    
    for i in range(num_requests):
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict",
                json=TEST_TOKEN,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            if response.status_code == 200:
                successful_requests += 1
            print(f"   Запрос {i+1}: {'✅' if response.status_code == 200 else '❌'}")
        except Exception as e:
            print(f"   Запрос {i+1}: ❌ {e}")
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_requests
    
    print(f"\n📊 Результаты производительности:")
    print(f"   Всего запросов: {num_requests}")
    print(f"   Успешных: {successful_requests}")
    print(f"   Общее время: {total_time:.2f}s")
    print(f"   Среднее время на запрос: {avg_time:.2f}s")
    print(f"   Запросов в секунду: {num_requests/total_time:.2f}")

def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ API ПРЕДСКАЗАНИЯ МЕМТОКЕНОВ")
    print("=" * 60)
    
    tests = [
        ("Health Check", test_health_check),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("Detailed Analysis", test_detailed_analysis),
        ("Feature Extraction", test_feature_extraction),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{'='*60}")
        result = test_func()
        results.append((test_name, result))
        time.sleep(1)  # Небольшая пауза между тестами
    
    # Тест производительности
    print(f"\n{'='*60}")
    run_performance_test()
    
    # Итоговые результаты
    print(f"\n{'='*60}")
    print("📊 ИТОГОВЫЕ РЕЗУЛЬТАТЫ ТЕСТОВ")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nОбщий результат: {passed}/{len(tests)} тестов прошли успешно")
    
    if passed == len(tests):
        print("🎉 Все тесты прошли успешно! API готово к использованию.")
    else:
        print("⚠️ Некоторые тесты не прошли. Проверьте логи для диагностики.")

if __name__ == "__main__":
    main()
