"""
Flask веб-приложение для гибридной модели (FT-Transformer + CatBoost)
Включает SHAP-визуализацию и генерацию финансовых предложений
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import shap
from io import BytesIO
import base64
from catboost import CatBoostRegressor

from model import FTTransformer
from train_hybrid import FTTransformerEmbedder, preprocess_data, USELESS_FEATURES

app = Flask(__name__)

# Глобальные переменные
embedder = None
catboost_model = None
device = None
feature_info = None
encoders = None
scaler = None


def load_hybrid_model(embedder_path='hybrid_model_embedder.pth', catboost_path='hybrid_model.cbm'):
    """Загрузка гибридной модели"""
    global embedder, catboost_model, device, feature_info, encoders, scaler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка embedder
    checkpoint = torch.load(embedder_path, map_location=device, weights_only=False)

    feature_info = checkpoint['feature_info']
    encoders = checkpoint['encoders']
    scaler = checkpoint['scaler']

    # Создаем базовую модель
    base_model = FTTransformer(
        n_num_features=feature_info['n_num_features'],
        cat_cardinalities=feature_info['cat_cardinalities'] if feature_info['n_cat_features'] > 0 else [],
        d_token=192,
        n_layers=3,
        n_heads=8,
        d_ffn=512,
        dropout=0.2,
        attention_dropout=0.3
    ).to(device)

    # Создаем embedder
    embedder = FTTransformerEmbedder(base_model)
    embedder.load_state_dict(checkpoint['embedder_state_dict'])
    embedder.eval()

    # Загрузка CatBoost
    catboost_model = CatBoostRegressor()
    catboost_model.load_model(catboost_path)

    print(f"✓ Гибридная модель загружена")
    print(f"  Embedder: {embedder_path}")
    print(f"  CatBoost: {catboost_path}")
    print(f"  Устройство: {device}")


def predict_income(client_data):
    """
    Предсказание дохода для клиента

    Args:
        client_data: dict с данными клиента

    Returns:
        predicted_income: float, предсказанный доход
    """
    if embedder is None or catboost_model is None:
        raise ValueError("Модель не загружена!")

    # Создаем DataFrame с недостающими признаками
    # Заполняем все признаки из обучающей выборки
    full_data = {}

    # Числовые признаки - заполняем нулями если не указаны
    for feature in feature_info['num_feature_names']:
        full_data[feature] = client_data.get(feature, 0)

    # Категориальные признаки - заполняем "MISSING" если не указаны
    for feature in feature_info['cat_feature_names']:
        full_data[feature] = client_data.get(feature, "MISSING")

    df = pd.DataFrame([full_data])

    # Предобработка
    X_num, X_cat, _, _, _, _, _ = preprocess_data(
        df,
        is_train=False,
        encoders=encoders,
        scaler=scaler,
        cat_feature_names=feature_info['cat_feature_names']
    )

    # Извлекаем эмбеддинги
    x_num = torch.FloatTensor(X_num).to(device) if X_num is not None else None
    x_cat = torch.LongTensor(X_cat).to(device) if X_cat is not None else None

    with torch.no_grad():
        embeddings = embedder(x_num, x_cat).cpu().numpy()

    # Комбинируем с исходными признаками
    X_combined = np.concatenate([X_num, embeddings], axis=1)

    # Предсказание через CatBoost
    prediction_log = catboost_model.predict(X_combined)[0]
    prediction = np.expm1(prediction_log)
    prediction = max(0, prediction)

    return prediction


def generate_financial_offers(predicted_income, client_data):
    """
    Генерация персонализированных финансовых предложений
    """
    offers = []

    # Категоризация дохода
    if predicted_income < 30000:
        income_category = "низкий"
        max_credit = predicted_income * 3
        max_card_limit = 50000
        deposit_interest = 5.5
    elif predicted_income < 80000:
        income_category = "средний"
        max_credit = predicted_income * 5
        max_card_limit = 150000
        deposit_interest = 6.0
    else:
        income_category = "высокий"
        max_credit = predicted_income * 8
        max_card_limit = 500000
        deposit_interest = 6.5

    # Предложение 1: Кредит
    offers.append({
        "type": "Потребительский кредит",
        "title": f"Кредит до {max_credit:,.0f} ₽",
        "description": f"Персональное предложение для клиентов с доходом {income_category} уровня",
        "interest_rate": "от 9.9%",
        "term": "до 5 лет",
        "icon": "💰",
        "details": {
            "Максимальная сумма": f"{max_credit:,.0f} ₽",
            "Ежемесячный платеж": f"≈ {max_credit * 0.02:,.0f} ₽",
            "Решение": "за 1 минуту"
        }
    })

    # Предложение 2: Кредитная карта
    offers.append({
        "type": "Кредитная карта",
        "title": f"Кредитный лимит до {max_card_limit:,.0f} ₽",
        "description": "100 дней без процентов на покупки",
        "interest_rate": "0% на 100 дней",
        "term": "бессрочно",
        "icon": "💳",
        "details": {
            "Кредитный лимит": f"до {max_card_limit:,.0f} ₽",
            "Кэшбэк": "до 10% за покупки",
            "Обслуживание": "0 ₽ при обороте от 10,000 ₽"
        }
    })

    # Предложение 3: Вклад
    offers.append({
        "type": "Накопительный счет",
        "title": f"Ставка {deposit_interest}% годовых",
        "description": "Сохраните и приумножьте свои накопления",
        "interest_rate": f"{deposit_interest}% годовых",
        "term": "без ограничений",
        "icon": "🏦",
        "details": {
            "Ставка": f"{deposit_interest}% годовых",
            "Пополнение": "в любое время",
            "Снятие": "без ограничений",
            "Страхование": "до 1,400,000 ₽"
        }
    })

    # Предложение 4: Инвестиции (для высокого дохода)
    if predicted_income >= 80000:
        offers.append({
            "type": "Инвестиции",
            "title": "Индивидуальный инвестиционный счет (ИИС)",
            "description": "Налоговый вычет до 52,000 ₽ в год",
            "interest_rate": "потенциальная доходность 10-15%",
            "term": "от 3 лет",
            "icon": "📈",
            "details": {
                "Налоговый вычет": "13% от внесенной суммы",
                "Минимальный взнос": "от 1,000 ₽",
                "Комиссия": "от 0%",
                "Доступ": "к акциям, облигациям, фондам"
            }
        })

    return offers


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint для предсказания"""
    try:
        # Получаем данные от клиента
        client_data = request.json

        if not client_data:
            return jsonify({'error': 'Нет данных'}), 400

        # Предсказание дохода
        predicted_income = predict_income(client_data)

        # Генерация финансовых предложений
        offers = generate_financial_offers(predicted_income, client_data)

        # Формируем ответ
        response = {
            'predicted_income': float(predicted_income),
            'predicted_income_formatted': f"{predicted_income:,.0f} ₽",
            'offers': offers,
            'model_type': 'Hybrid (FT-Transformer + CatBoost)'
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/features')
def get_features():
    """Get available features"""
    if feature_info is None:
        return jsonify({'error': 'Model not loaded'}), 500

    return jsonify({
        'num_features': feature_info['num_feature_names'],
        'cat_features': feature_info['cat_feature_names'],
        'total': feature_info['n_num_features'] + feature_info['n_cat_features']
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'embedder_loaded': embedder is not None,
        'catboost_loaded': catboost_model is not None,
        'device': str(device) if device else None
    })


def create_templates():
    """Создание HTML шаблонов"""
    os.makedirs('templates', exist_ok=True)

    index_html = """
<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Альфа-Банк - Прогноз доходов (Hybrid Model)</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }

        .container {
            max-width: 1200px;
            margin: 0 auto;
        }

        .header {
            text-align: center;
            color: white;
            margin-bottom: 40px;
        }

        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }

        .header p {
            font-size: 1.2em;
            opacity: 0.9;
        }

        .badge {
            display: inline-block;
            background: rgba(255,255,255,0.2);
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-top: 10px;
        }

        .card {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            margin-bottom: 30px;
        }

        .form-group {
            margin-bottom: 20px;
        }

        label {
            display: block;
            margin-bottom: 5px;
            font-weight: 600;
            color: #333;
        }

        input, select, textarea {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s;
        }

        input:focus, select:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }

        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            font-size: 18px;
            border-radius: 8px;
            cursor: pointer;
            width: 100%;
            transition: transform 0.2s;
        }

        .btn:hover {
            transform: translateY(-2px);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .result {
            display: none;
            margin-top: 30px;
        }

        .result.show {
            display: block;
            animation: fadeIn 0.5s;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .income-display {
            text-align: center;
            padding: 30px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 15px;
            color: white;
            margin-bottom: 30px;
        }

        .income-display h2 {
            font-size: 1.5em;
            margin-bottom: 10px;
        }

        .income-display .amount {
            font-size: 3em;
            font-weight: bold;
        }

        .model-info {
            text-align: center;
            margin-top: 10px;
            font-size: 0.9em;
            opacity: 0.9;
        }

        .offers {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-top: 30px;
        }

        .offer-card {
            background: #f8f9fa;
            border-radius: 12px;
            padding: 20px;
            transition: transform 0.3s, box-shadow 0.3s;
        }

        .offer-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }

        .offer-icon {
            font-size: 3em;
            margin-bottom: 10px;
        }

        .offer-title {
            font-size: 1.3em;
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }

        .offer-description {
            color: #666;
            margin-bottom: 15px;
        }

        .offer-details {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
        }

        .offer-detail-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #e0e0e0;
        }

        .offer-detail-item:last-child {
            border-bottom: none;
        }

        .loader {
            display: none;
            text-align: center;
            padding: 20px;
        }

        .loader.show {
            display: block;
        }

        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .error {
            background: #f8d7da;
            color: #721c24;
            padding: 15px;
            border-radius: 8px;
            margin-top: 20px;
            display: none;
        }

        .error.show {
            display: block;
        }

        .info-box {
            background: #e7f3ff;
            border-left: 4px solid #2196F3;
            padding: 15px;
            margin-bottom: 20px;
            border-radius: 4px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🏦 Альфа-Банк</h1>
            <p>Прогнозирование доходов клиентов с помощью AI</p>
            <span class="badge">🔥 Hybrid Model: FT-Transformer + CatBoost</span>
        </div>

        <div class="card">
            <div class="info-box">
                <strong>ℹ️ О модели:</strong> Используется гибридный подход - FT-Transformer извлекает глубокие эмбеддинги,
                а CatBoost строит финальное предсказание. Это предотвращает переобучение и дает лучшее качество!
            </div>

            <h2>Введите данные клиента (JSON)</h2>

            <div style="margin-bottom: 20px;">
                <button type="button" class="btn" onclick="loadFeatureList()" style="background: #2196F3; width: auto; display: inline-block; padding: 10px 20px;">
                    📋 Показать список всех признаков
                </button>
                <button type="button" class="btn" onclick="loadMinimalExample()" style="background: #4CAF50; width: auto; display: inline-block; padding: 10px 20px; margin-left: 10px;">
                    ⚡ Загрузить минимальный пример
                </button>
            </div>

            <div id="featureInfo" style="display: none; background: #f5f5f5; padding: 15px; border-radius: 8px; margin-bottom: 20px; max-height: 200px; overflow-y: auto;">
                <strong>Доступные признаки:</strong>
                <div id="featureList" style="margin-top: 10px; font-size: 0.9em;"></div>
            </div>

            <form id="predictionForm">
                <div class="form-group">
                    <label for="jsonData">Данные клиента в формате JSON:</label>
                    <textarea id="jsonData" rows="12" placeholder='{"feature1": value1, "feature2": value2, ...}'></textarea>
                    <small style="color: #666;">
                        💡 Совет: Можно указать только важные признаки, остальные заполнятся автоматически нулями/MISSING
                    </small>
                </div>

                <button type="submit" class="btn" id="submitBtn">
                    🚀 Рассчитать доход
                </button>
            </form>

            <div class="loader" id="loader">
                <div class="spinner"></div>
                <p>Анализируем данные через гибридную модель...</p>
            </div>

            <div class="error" id="error"></div>
        </div>

        <div id="results" class="result">
            <div class="card">
                <div class="income-display">
                    <h2>Предсказанный доход клиента</h2>
                    <div class="amount" id="incomeAmount">0 ₽</div>
                    <div class="model-info" id="modelInfo">Hybrid Model</div>
                </div>

                <h2 style="margin-top: 40px; margin-bottom: 20px;">💼 Персонализированные финансовые предложения</h2>
                <div class="offers" id="offers"></div>
            </div>
        </div>
    </div>

    <script>
        // Минимальный пример данных
        const minimalExample = {
            "info_note": "Можно указать только несколько признаков, остальные заполнятся автоматически",
            "feature_example_1": 100,
            "feature_example_2": "VALUE"
        };

        // Загрузить список всех признаков
        async function loadFeatureList() {
            const featureInfo = document.getElementById('featureInfo');
            const featureList = document.getElementById('featureList');

            featureInfo.style.display = 'block';
            featureList.innerHTML = 'Загрузка...';

            try {
                const response = await fetch('/features');
                const data = await response.json();

                let html = '<div style="margin-bottom: 10px;"><strong>Всего признаков: ' + data.total + '</strong></div>';
                html += '<div><strong>Числовые (' + data.num_features.length + '):</strong><br>';
                html += data.num_features.slice(0, 20).join(', ');
                if (data.num_features.length > 20) {
                    html += ', ... (и еще ' + (data.num_features.length - 20) + ')';
                }
                html += '</div><br>';
                html += '<div><strong>Категориальные (' + data.cat_features.length + '):</strong><br>';
                html += data.cat_features.join(', ');
                html += '</div>';

                featureList.innerHTML = html;
            } catch (e) {
                featureList.innerHTML = 'Ошибка загрузки: ' + e.message;
            }
        }

        // Загрузить минимальный пример
        function loadMinimalExample() {
            document.getElementById('jsonData').value = JSON.stringify(minimalExample, null, 2);
        }

        // При загрузке страницы - вставляем пример
        document.addEventListener('DOMContentLoaded', function() {
            loadMinimalExample();
        });

        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const jsonData = document.getElementById('jsonData').value;

            if (!jsonData.trim()) {
                showError('Пожалуйста, введите данные клиента в формате JSON');
                return;
            }

            let data;
            try {
                data = JSON.parse(jsonData);
            } catch (e) {
                showError('Ошибка парсинга JSON: ' + e.message);
                return;
            }

            // Показываем loader
            document.getElementById('loader').classList.add('show');
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('results').classList.remove('show');
            document.getElementById('error').classList.remove('show');

            try {
                // Отправляем запрос
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify(data)
                });

                if (!response.ok) {
                    throw new Error('Ошибка при предсказании');
                }

                const result = await response.json();

                // Показываем результаты
                displayResults(result);

            } catch (error) {
                showError('Ошибка: ' + error.message);
            } finally {
                document.getElementById('loader').classList.remove('show');
                document.getElementById('submitBtn').disabled = false;
            }
        });

        function displayResults(result) {
            // Доход
            document.getElementById('incomeAmount').textContent = result.predicted_income_formatted;
            document.getElementById('modelInfo').textContent = result.model_type || 'Hybrid Model';

            // Предложения
            const offersHtml = result.offers.map(offer => `
                <div class="offer-card">
                    <div class="offer-icon">${offer.icon}</div>
                    <div class="offer-title">${offer.title}</div>
                    <div class="offer-description">${offer.description}</div>
                    <div style="color: #667eea; font-weight: bold;">${offer.interest_rate}</div>
                    <div class="offer-details">
                        ${Object.entries(offer.details).map(([key, value]) => `
                            <div class="offer-detail-item">
                                <span>${key}</span>
                                <strong>${value}</strong>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('');

            document.getElementById('offers').innerHTML = offersHtml;

            // Показываем результаты
            document.getElementById('results').classList.add('show');

            // Прокручиваем к результатам
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.classList.add('show');
        }
    </script>
</body>
</html>
"""

    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)

    print("✓ HTML templates created")


if __name__ == '__main__':
    import sys

    # Создаем templates
    create_templates()

    # Загружаем модель
    embedder_path = 'hybrid_model_embedder.pth'
    catboost_path = 'hybrid_model.cbm'

    if os.path.exists(embedder_path) and os.path.exists(catboost_path):
        print(f"Загрузка гибридной модели...")
        load_hybrid_model(embedder_path, catboost_path)
        print("✓ Модель успешно загружена!")
    else:
        print(f"⚠️  ПРЕДУПРЕЖДЕНИЕ: Модель не найдена!")
        print(f"  Ищу: {embedder_path} и {catboost_path}")
        print(f"\nСначала обучите модель:")
        print(f"  ./quick_train.sh")
        sys.exit(1)

    # Запускаем приложение
    print("\n" + "="*60)
    print("Запуск Flask приложения...")
    print("="*60)
    print("\n🌐 Сервер запущен на порту 5000")
    print("\n📡 Для доступа с локального компьютера:")
    print("   1. Откройте НОВЫЙ терминал на СВОЕМ компьютере (не в SSH!)")
    print("   2. Выполните команду:")
    print("      ssh -L 5000:localhost:5000 -p 10210 shkodnik1917@proxy2.cod.phystech.edu")
    print("   3. Введите пароль от SSH")
    print("   4. Откройте браузер: http://localhost:5000")
    print("\n" + "="*60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
