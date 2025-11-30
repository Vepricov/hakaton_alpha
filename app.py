"""
Flask веб-приложение для гибридной модели (FT-Transformer + CatBoost)
Включает выбор клиента по ID из тестовой выборки
Дизайн в стиле Альфа-Банка
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from flask import Flask, render_template, request, jsonify
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
test_data = None


def load_hybrid_model(embedder_path='hybrid_model_embedder.pth', catboost_path='hybrid_model.cbm'):
    """Загрузка гибридной модели"""
    global embedder, catboost_model, device, feature_info, encoders, scaler

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка checkpoint
    checkpoint = torch.load(embedder_path, map_location=device, weights_only=False)

    feature_info = checkpoint['feature_info']
    encoders = checkpoint['encoders']
    scaler = checkpoint['scaler']

    # Загружаем конфигурацию модели
    if 'model_config' in checkpoint and checkpoint['model_config'] is not None:
        # Используем сохраненную конфигурацию
        model_config = checkpoint['model_config']
        print(f"✓ Загружена конфигурация модели из checkpoint")
    else:
        # Пробуем загрузить из JSON файла
        config_path = embedder_path.replace('_embedder.pth', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            print(f"✓ Загружена конфигурация модели из {config_path}")
        else:
            # Fallback: определяем из state_dict
            print(f"⚠️  Конфигурация не найдена, определяю из state_dict...")
            state_dict = checkpoint['embedder_state_dict']
            max_layer = max([int(k.split('.')[1]) for k in state_dict.keys() if 'transformer_blocks.' in k])
            n_embeddings = checkpoint.get('n_embeddings', 192)

            model_config = {
                'n_num_features': feature_info['n_num_features'],
                'cat_cardinalities': feature_info['cat_cardinalities'] if feature_info['n_cat_features'] > 0 else [],
                'd_token': n_embeddings,
                'n_layers': max_layer + 1,
                'n_heads': 8,
                'd_ffn': 512,
                'dropout': 0.2,
                'attention_dropout': 0.3
            }

    # Создаем базовую FT-Transformer модель с правильной конфигурацией
    base_model = FTTransformer(**model_config).to(device)

    # Оборачиваем в embedder и загружаем веса
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
    print(f"  Архитектура: {model_config['n_layers']} layers, {model_config['d_token']} d_token")


def load_test_data(test_path='hackathon_income_test.csv'):
    """Загрузка тестовой выборки"""
    global test_data

    test_data = pd.read_csv(test_path, sep=';', decimal=',')
    print(f"✓ Тестовая выборка загружена: {len(test_data)} записей")


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

    # Создаем DataFrame из данных клиента
    df = pd.DataFrame([client_data])

    # Выравниваем колонки с обучающей выборкой
    # Добавляем недостающие числовые признаки (но НЕ затираем существующие!)
    for feature in feature_info['num_feature_names']:
        if feature not in df.columns:
            # Добавляем отсутствующий признак
            df[feature] = 0.0
        elif df[feature].dtype == 'object':
            # Конвертируем object в числовой тип
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0).astype(float)
        else:
            # Просто заполняем NaN нулями, остальное оставляем как есть
            df[feature] = df[feature].fillna(0).astype(float)

    # Добавляем недостающие категориальные признаки
    for feature in feature_info['cat_feature_names']:
        if feature not in df.columns:
            df[feature] = "MISSING"
        else:
            df[feature] = df[feature].fillna("MISSING").astype(str)

    # Удаляем target и w если они есть (чтобы не мешали)
    df = df.drop(columns=['target', 'w'], errors='ignore')

    # Предобработка с is_train=False (чтобы использовать переданный scaler)
    X_num, X_cat, _, _, _, _, _ = preprocess_data(
        df,
        is_train=False,  # False чтобы использовать переданный scaler
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
        data = request.json

        if not data:
            return jsonify({'error': 'Нет данных'}), 400

        # Проверяем, передан ли ID
        if 'client_id' in data:
            # Получаем данные из тестовой выборки
            client_id = int(data['client_id'])
            if test_data is None:
                return jsonify({'error': 'Тестовая выборка не загружена'}), 500

            if client_id not in test_data['id'].values:
                return jsonify({'error': f'Клиент с ID {client_id} не найден'}), 404

            # Получаем данные клиента
            client_row = test_data[test_data['id'] == client_id].iloc[0]
            client_data = client_row.to_dict()
        else:
            client_data = data

        # Предсказание дохода
        predicted_income = predict_income(client_data)

        # Генерация финансовых предложений
        offers = generate_financial_offers(predicted_income, client_data)

        # Формируем ответ
        response = {
            'predicted_income': float(predicted_income),
            'predicted_income_formatted': f"{predicted_income:,.0f} ₽",
            'offers': offers,
            'model_type': 'Hybrid (FT-Transformer + CatBoost)',
            'client_data': {k: str(v) for k, v in list(client_data.items())[:10]}  # Первые 10 параметров для отображения
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/client/<int:client_id>')
def get_client(client_id):
    """Получить данные клиента по ID"""
    try:
        if test_data is None:
            return jsonify({'error': 'Тестовая выборка не загружена'}), 500

        if client_id not in test_data['id'].values:
            return jsonify({'error': f'Клиент с ID {client_id} не найден'}), 404

        client_row = test_data[test_data['id'] == client_id].iloc[0]
        client_data = client_row.to_dict()

        # Конвертируем все значения в строки для JSON
        client_data_serializable = {k: str(v) if pd.notna(v) else None for k, v in client_data.items()}

        return jsonify(client_data_serializable)

    except Exception as e:
        print(f"Error getting client: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/clients/random')
def get_random_clients():
    """Получить случайные ID клиентов"""
    try:
        if test_data is None:
            return jsonify({'error': 'Тестовая выборка не загружена'}), 500

        # Получаем 10 случайных ID
        random_ids = test_data['id'].sample(n=min(10, len(test_data))).tolist()

        return jsonify({
            'ids': random_ids,
            'total_clients': len(test_data)
        })

    except Exception as e:
        print(f"Error getting random clients: {e}")
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
        'test_data_loaded': test_data is not None,
        'test_data_size': len(test_data) if test_data is not None else 0,
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
    <title>Альфа-Банк — AI Прогноз Доходов</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', Roboto, -apple-system, BlinkMacSystemFont, sans-serif;
            background: #FFFFFF;
            min-height: 100vh;
            color: #1A1A1A;
        }

        /* Header с красным фоном Альфа-Банка */
        .header {
            background: linear-gradient(135deg, #EF3124 0%, #C41E3A 100%);
            color: white;
            padding: 40px 20px;
            box-shadow: 0 4px 20px rgba(239, 49, 36, 0.3);
        }

        .header-content {
            max-width: 1200px;
            margin: 0 auto;
        }

        .logo {
            font-size: 2.5em;
            font-weight: 700;
            margin-bottom: 10px;
            letter-spacing: -1px;
        }

        .tagline {
            font-size: 1.3em;
            opacity: 0.95;
            font-weight: 300;
        }

        .badge {
            display: inline-block;
            background: rgba(255, 255, 255, 0.2);
            backdrop-filter: blur(10px);
            padding: 8px 20px;
            border-radius: 20px;
            font-size: 0.85em;
            margin-top: 15px;
            border: 1px solid rgba(255, 255, 255, 0.3);
        }

        .container {
            max-width: 1200px;
            margin: -30px auto 40px;
            padding: 0 20px;
        }

        /* Карточки с тенью */
        .card {
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
            margin-bottom: 30px;
            border: 1px solid #F0F0F0;
        }

        .card h2 {
            color: #1A1A1A;
            margin-bottom: 25px;
            font-size: 1.8em;
            font-weight: 600;
        }

        /* Секция выбора клиента */
        .client-selector {
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
        }

        .input-group {
            display: flex;
            gap: 15px;
            align-items: flex-end;
        }

        .form-group {
            flex: 1;
        }

        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
            font-size: 0.95em;
        }

        input[type="number"], input[type="text"] {
            width: 100%;
            padding: 15px 20px;
            border: 2px solid #E0E0E0;
            border-radius: 12px;
            font-size: 16px;
            transition: all 0.3s;
            font-family: inherit;
        }

        input:focus {
            outline: none;
            border-color: #EF3124;
            box-shadow: 0 0 0 4px rgba(239, 49, 36, 0.1);
        }

        /* Кнопки в стиле Альфа-Банка */
        .btn {
            background: linear-gradient(135deg, #EF3124 0%, #C41E3A 100%);
            color: white;
            border: none;
            padding: 16px 32px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 4px 15px rgba(239, 49, 36, 0.3);
            white-space: nowrap;
        }

        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(239, 49, 36, 0.4);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }

        .btn-secondary {
            background: #1A1A1A;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .btn-secondary:hover {
            background: #333;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }

        /* Случайные ID */
        .random-ids {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            margin-top: 15px;
        }

        .id-chip {
            background: #F5F5F5;
            padding: 8px 16px;
            border-radius: 20px;
            cursor: pointer;
            transition: all 0.2s;
            border: 2px solid transparent;
            font-weight: 500;
        }

        .id-chip:hover {
            background: #EF3124;
            color: white;
            transform: translateY(-2px);
            box-shadow: 0 4px 10px rgba(239, 49, 36, 0.3);
        }

        /* Информационный блок */
        .info-box {
            background: linear-gradient(135deg, #FFF5F5 0%, #FFE8E8 100%);
            border-left: 4px solid #EF3124;
            padding: 20px;
            margin-bottom: 25px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            gap: 15px;
        }

        .info-icon {
            font-size: 2em;
            flex-shrink: 0;
        }

        /* Loader */
        .loader {
            display: none;
            text-align: center;
            padding: 40px;
        }

        .loader.show {
            display: block;
        }

        .spinner {
            border: 4px solid #F0F0F0;
            border-top: 4px solid #EF3124;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            animation: spin 0.8s linear infinite;
            margin: 0 auto 20px;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loader-text {
            color: #666;
            font-size: 1.1em;
        }

        /* Результаты */
        .result {
            display: none;
        }

        .result.show {
            display: block;
            animation: fadeInUp 0.5s;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(30px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Отображение дохода */
        .income-display {
            text-align: center;
            padding: 50px 30px;
            background: linear-gradient(135deg, #EF3124 0%, #C41E3A 100%);
            border-radius: 20px;
            color: white;
            margin-bottom: 40px;
            position: relative;
            overflow: hidden;
        }

        .income-display::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: pulse 3s ease-in-out infinite;
        }

        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 0.5; }
            50% { transform: scale(1.1); opacity: 0.8; }
        }

        .income-display h2 {
            font-size: 1.3em;
            margin-bottom: 15px;
            color: white;
            font-weight: 400;
            position: relative;
            z-index: 1;
        }

        .income-amount {
            font-size: 3.5em;
            font-weight: 700;
            margin-bottom: 10px;
            position: relative;
            z-index: 1;
            text-shadow: 0 2px 10px rgba(0, 0, 0, 0.2);
        }

        .model-info {
            font-size: 0.95em;
            opacity: 0.9;
            position: relative;
            z-index: 1;
        }

        /* Информация о клиенте */
        .client-info {
            background: #F8F8F8;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 30px;
        }

        .client-info h3 {
            margin-bottom: 15px;
            color: #1A1A1A;
        }

        .client-params {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 10px;
        }

        .param-item {
            background: white;
            padding: 10px 15px;
            border-radius: 8px;
            font-size: 0.9em;
        }

        .param-label {
            color: #666;
            font-weight: 500;
        }

        .param-value {
            color: #1A1A1A;
            font-weight: 600;
            margin-left: 5px;
        }

        /* Предложения */
        .offers {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
        }

        .offer-card {
            background: linear-gradient(135deg, #FFFFFF 0%, #F8F8F8 100%);
            border-radius: 16px;
            padding: 30px;
            transition: all 0.3s;
            border: 2px solid #F0F0F0;
            position: relative;
            overflow: hidden;
        }

        .offer-card::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #EF3124 0%, #C41E3A 100%);
            transform: scaleX(0);
            transition: transform 0.3s;
        }

        .offer-card:hover {
            transform: translateY(-8px);
            box-shadow: 0 15px 40px rgba(239, 49, 36, 0.15);
            border-color: #EF3124;
        }

        .offer-card:hover::before {
            transform: scaleX(1);
        }

        .offer-icon {
            font-size: 3.5em;
            margin-bottom: 15px;
            display: block;
        }

        .offer-title {
            font-size: 1.4em;
            font-weight: 700;
            color: #1A1A1A;
            margin-bottom: 12px;
        }

        .offer-description {
            color: #666;
            margin-bottom: 15px;
            line-height: 1.5;
        }

        .offer-rate {
            color: #EF3124;
            font-weight: 700;
            font-size: 1.1em;
            margin-bottom: 20px;
        }

        .offer-details {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            border: 1px solid #E8E8E8;
        }

        .offer-detail-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #F0F0F0;
        }

        .offer-detail-item:last-child {
            border-bottom: none;
        }

        .detail-label {
            color: #666;
            font-size: 0.9em;
        }

        .detail-value {
            color: #1A1A1A;
            font-weight: 600;
            text-align: right;
        }

        /* Error */
        .error {
            background: #FFF0F0;
            color: #C41E3A;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
            display: none;
            border-left: 4px solid #EF3124;
        }

        .error.show {
            display: block;
            animation: shake 0.5s;
        }

        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-10px); }
            75% { transform: translateX(10px); }
        }

        /* Responsive */
        @media (max-width: 768px) {
            .header {
                padding: 30px 20px;
            }

            .logo {
                font-size: 2em;
            }

            .tagline {
                font-size: 1.1em;
            }

            .card {
                padding: 25px;
            }

            .input-group {
                flex-direction: column;
            }

            .btn {
                width: 100%;
            }

            .income-amount {
                font-size: 2.5em;
            }

            .offers {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">Альфа-Банк</div>
            <div class="tagline">AI-прогнозирование доходов клиентов</div>
            <span class="badge">🚀 Hybrid Model: FT-Transformer + CatBoost</span>
        </div>
    </div>

    <div class="container">
        <div class="card">
            <div class="info-box">
                <div class="info-icon">🎯</div>
                <div>
                    <strong>Как это работает:</strong> Выберите клиента по ID из тестовой выборки, и наша гибридная модель
                    (FT-Transformer + CatBoost) предскажет его доход и сформирует персонализированные финансовые предложения.
                </div>
            </div>

            <h2>Выберите клиента</h2>

            <form id="predictionForm" class="client-selector">
                <div class="input-group">
                    <div class="form-group">
                        <label for="clientId">🆔 ID клиента</label>
                        <input type="number" id="clientId" placeholder="Введите ID клиента" required>
                    </div>
                    <button type="submit" class="btn" id="submitBtn">
                        📊 Рассчитать доход
                    </button>
                </div>

                <div>
                    <button type="button" class="btn btn-secondary" onclick="loadRandomIds()" style="width: auto;">
                        🎲 Загрузить случайные ID
                    </button>
                    <div class="random-ids" id="randomIds"></div>
                </div>
            </form>

            <div class="loader" id="loader">
                <div class="spinner"></div>
                <div class="loader-text">Анализируем данные клиента...</div>
            </div>

            <div class="error" id="error"></div>
        </div>

        <div id="results" class="result">
            <div class="card">
                <div class="income-display">
                    <h2>Прогнозируемый доход клиента</h2>
                    <div class="income-amount" id="incomeAmount">0 ₽</div>
                    <div class="model-info" id="modelInfo">Hybrid Model</div>
                </div>

                <div class="client-info" id="clientInfo" style="display: none;">
                    <h3>📋 Основные параметры клиента</h3>
                    <div class="client-params" id="clientParams"></div>
                </div>

                <h2>💼 Персонализированные финансовые предложения</h2>
                <div class="offers" id="offers"></div>
            </div>
        </div>
    </div>

    <script>
        // Загрузить случайные ID при загрузке страницы
        document.addEventListener('DOMContentLoaded', function() {
            loadRandomIds();
        });

        // Загрузить случайные ID клиентов
        async function loadRandomIds() {
            try {
                const response = await fetch('/clients/random');
                const data = await response.json();

                const container = document.getElementById('randomIds');
                container.innerHTML = data.ids.map(id =>
                    `<div class="id-chip" onclick="selectClient(${id})">ID: ${id}</div>`
                ).join('');

            } catch (error) {
                console.error('Error loading random IDs:', error);
            }
        }

        // Выбрать клиента по клику на ID
        function selectClient(id) {
            document.getElementById('clientId').value = id;
            // Автоматически отправляем форму
            document.getElementById('predictionForm').dispatchEvent(new Event('submit'));
        }

        // Обработка формы
        document.getElementById('predictionForm').addEventListener('submit', async (e) => {
            e.preventDefault();

            const clientId = document.getElementById('clientId').value;

            if (!clientId) {
                showError('Пожалуйста, введите ID клиента');
                return;
            }

            // Показываем loader
            document.getElementById('loader').classList.add('show');
            document.getElementById('submitBtn').disabled = true;
            document.getElementById('results').classList.remove('show');
            document.getElementById('error').classList.remove('show');

            try {
                // Отправляем запрос на предсказание
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ client_id: clientId })
                });

                if (!response.ok) {
                    const errorData = await response.json();
                    throw new Error(errorData.error || 'Ошибка при предсказании');
                }

                const result = await response.json();

                // Показываем результаты
                displayResults(result);

            } catch (error) {
                showError(error.message);
            } finally {
                document.getElementById('loader').classList.remove('show');
                document.getElementById('submitBtn').disabled = false;
            }
        });

        function displayResults(result) {
            // Доход
            document.getElementById('incomeAmount').textContent = result.predicted_income_formatted;
            document.getElementById('modelInfo').textContent = result.model_type || 'Hybrid Model';

            // Информация о клиенте
            if (result.client_data) {
                const clientInfo = document.getElementById('clientInfo');
                const clientParams = document.getElementById('clientParams');

                clientParams.innerHTML = Object.entries(result.client_data).map(([key, value]) => `
                    <div class="param-item">
                        <span class="param-label">${key}:</span>
                        <span class="param-value">${value}</span>
                    </div>
                `).join('');

                clientInfo.style.display = 'block';
            }

            // Предложения
            const offersHtml = result.offers.map(offer => `
                <div class="offer-card">
                    <div class="offer-icon">${offer.icon}</div>
                    <div class="offer-title">${offer.title}</div>
                    <div class="offer-description">${offer.description}</div>
                    <div class="offer-rate">${offer.interest_rate}</div>
                    <div class="offer-details">
                        ${Object.entries(offer.details).map(([key, value]) => `
                            <div class="offer-detail-item">
                                <span class="detail-label">${key}</span>
                                <span class="detail-value">${value}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `).join('');

            document.getElementById('offers').innerHTML = offersHtml;

            // Показываем результаты
            document.getElementById('results').classList.add('show');

            // Прокручиваем к результатам
            setTimeout(() => {
                document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }

        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = '❌ ' + message;
            errorDiv.classList.add('show');

            setTimeout(() => {
                errorDiv.classList.remove('show');
            }, 5000);
        }
    </script>
</body>
</html>
"""

    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)

    print("✓ HTML templates created")


if __name__ == '__main__':
    # Создаем шаблоны
    create_templates()

    # Загружаем модель
    load_hybrid_model()

    # Загружаем тестовую выборку
    load_test_data()

    # Запускаем сервер
    print("\n" + "="*60)
    print("🚀 Запуск веб-приложения...")
    print("="*60)
    app.run(debug=True, host='0.0.0.0', port=5000)
