"""
Flask веб-приложение для гибридной модели (FT-Transformer + CatBoost)
С визуализацией важности признаков и интерпретацией факторов влияния
Дизайн в стиле Альфа-Банка
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')  # Для серверного рендеринга
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, request, jsonify
from catboost import CatBoostRegressor
import catboost

from model import FTTransformer
from train_hybrid import FTTransformerEmbedder, preprocess_data

app = Flask(__name__)

# Глобальные переменные
embedder = None
catboost_model = None
device = None
feature_info = None
encoders = None
scaler = None
test_data = None
feature_names_combined = None
feature_descriptions = None


def load_hybrid_model(embedder_path='hybrid_model_embedder.pth', catboost_path='hybrid_model.cbm'):
    """Загрузка гибридной модели"""
    global embedder, catboost_model, device, feature_info, encoders, scaler, feature_names_combined, feature_descriptions

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Загрузка checkpoint
    checkpoint = torch.load(embedder_path, map_location=device, weights_only=False)

    feature_info = checkpoint['feature_info']
    encoders = checkpoint['encoders']
    scaler = checkpoint['scaler']

    # Загружаем конфигурацию модели
    if 'model_config' in checkpoint and checkpoint['model_config'] is not None:
        model_config = checkpoint['model_config']
        print("✓ Загружена конфигурация модели из checkpoint")
    else:
        config_path = embedder_path.replace('_embedder.pth', '_config.json')
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            print(f"✓ Загружена конфигурация модели из {config_path}")
        else:
            print("⚠️  Конфигурация не найдена, определяю из state_dict...")
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

    # Создаем базовую FT-Transformer модель
    base_model = FTTransformer(**model_config).to(device)

    # Оборачиваем в embedder и загружаем веса
    embedder = FTTransformerEmbedder(base_model)
    embedder.load_state_dict(checkpoint['embedder_state_dict'])
    embedder.eval()

    # Загрузка CatBoost
    catboost_model = CatBoostRegressor()
    catboost_model.load_model(catboost_path)

    # Создаем имена признаков
    feature_names_combined = feature_info['num_feature_names'] + [f'embedding_{i}' for i in range(model_config['d_token'])]

    print("✓ Гибридная модель загружена")
    print(f"  Embedder: {embedder_path}")
    print(f"  CatBoost: {catboost_path}")
    print(f"  Устройство: {device}")
    print(f"  Архитектура: {model_config['n_layers']} layers, {model_config['d_token']} d_token")

    # Загружаем описания признаков
    try:
        desc_df = pd.read_csv('features_description.csv', sep=';', encoding='cp1251')
        feature_descriptions = dict(zip(desc_df.iloc[:, 0], desc_df.iloc[:, 1]))
        print(f"✓ Загружено описаний признаков: {len(feature_descriptions)}")
    except Exception as e:
        print(f"⚠️  Не удалось загрузить описания признаков: {e}")
        feature_descriptions = {}


def load_test_data(test_path='hackathon_income_test.csv'):
    """Загрузка тестовой выборки"""
    global test_data

    test_data = pd.read_csv(test_path, sep=';', decimal=',')
    print(f"✓ Тестовая выборка загружена: {len(test_data)} записей")


def predict_income_with_explanation(client_data):
    """
    Предсказание дохода для клиента с объяснением

    Returns:
        predicted_income: float
        top_features: dict с важностью признаков
        explanation: текстовое объяснение
    """
    if embedder is None or catboost_model is None:
        raise ValueError("Модель не загружена!")

    # Создаем DataFrame из данных клиента
    df = pd.DataFrame([client_data])

    # Выравниваем колонки с обучающей выборкой
    for feature in feature_info['num_feature_names']:
        if feature not in df.columns:
            df[feature] = 0.0
        elif df[feature].dtype == 'object':
            df[feature] = pd.to_numeric(df[feature], errors='coerce').fillna(0).astype(float)
        else:
            df[feature] = df[feature].fillna(0).astype(float)

    for feature in feature_info['cat_feature_names']:
        if feature not in df.columns:
            df[feature] = "MISSING"
        else:
            df[feature] = df[feature].fillna("MISSING").astype(str)

    df = df.drop(columns=['target', 'w'], errors='ignore')

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

    # Получаем SHAP values для конкретного клиента (локальная важность)
    try:
        shap_values = catboost_model.get_feature_importance(
            type='ShapValues',
            data=catboost.Pool(X_combined)
        )
        # shap_values возвращает матрицу [n_samples, n_features + 1]
        # Последний элемент - это базовое значение (bias)
        # Берем абсолютные значения для первого (единственного) примера
        client_shap = np.abs(shap_values[0, :-1])

        # Берем только числовые признаки (без эмбеддингов)
        num_features_count = len(feature_info['num_feature_names'])
        client_shap_num = client_shap[:num_features_count]

    except Exception as e:
        print(f"Warning: Could not get SHAP values, using global importance: {e}")
        # Fallback на глобальную важность
        feature_importance = catboost_model.get_feature_importance()
        client_shap_num = feature_importance[:len(feature_info['num_feature_names'])]

    # Находим топ-10 наиболее важных признаков для ЭТОГО клиента
    top_features_indices = np.argsort(client_shap_num)[-10:][::-1]

    # Берем важности только топ-признаков для нормализации
    top_importances = client_shap_num[top_features_indices]
    importance_sum = top_importances.sum()

    # Защита от деления на ноль
    if importance_sum == 0:
        importance_sum = 1.0

    top_features = []
    for idx in top_features_indices:
        feature_name = feature_info['num_feature_names'][idx]
        importance = client_shap_num[idx]
        value = X_num[0, idx]

        # Получаем исходное значение (до нормализации) если возможно
        original_value = client_data.get(feature_name, value)

        # Безопасная конвертация original_value с обработкой NaN
        if isinstance(original_value, (int, float)):
            if np.isnan(original_value):
                original_value_safe = 0.0
            else:
                original_value_safe = float(original_value)
        else:
            original_value_safe = str(original_value)

        # Генерируем пояснение для признака
        explanation = generate_feature_explanation(feature_name, original_value_safe)

        top_features.append({
            'name': feature_name,
            'importance': float(importance),
            'value': float(value) if not np.isnan(value) else 0.0,
            'original_value': original_value_safe,
            'normalized_importance': float(importance / importance_sum * 100),
            'explanation': explanation
        })

    return prediction, top_features


def generate_feature_explanation(feature_name, value):
    """Генерация пояснения для конкретного признака"""

    # Сначала пытаемся получить описание из загруженного словаря
    if feature_descriptions and feature_name in feature_descriptions:
        description = feature_descriptions[feature_name]
        # Добавляем контекстное пояснение на основе значения
        context = get_context_explanation(feature_name, value)
        if context:
            return f"{description}. {context}"
        return description

    # Fallback на базовые описания
    feature_display_names = {
        'Age': 'Возраст клиента',
        'age': 'Возраст клиента',
        'education': 'Уровень образования',
        'work_experience': 'Стаж работы',
        'salary': 'Текущая зарплата',
        'income': 'Текущий доход',
        'loan_amount': 'Сумма кредита',
        'credit_score': 'Кредитный рейтинг',
        'num_credits': 'Количество кредитов',
        'employment_type': 'Тип занятости',
    }

    display_name = feature_display_names.get(feature_name, feature_name.replace('_', ' ').title())

    # Контекстные пояснения
    context = get_context_explanation(feature_name, value)
    if context:
        return f"{display_name}. {context}"

    # Общее пояснение
    return f"{display_name}. Важный фактор для прогнозирования уровня дохода."


def get_context_explanation(feature_name, value):
    """Получить контекстное пояснение на основе значения признака"""

    if 'age' in feature_name.lower():
        try:
            age_val = float(value)
            if age_val < 25:
                return "Молодой возраст — начало карьеры, потенциал роста"
            elif age_val < 35:
                return "Оптимальный возраст для активного карьерного роста"
            elif age_val < 50:
                return "Зрелый возраст — устоявшаяся карьера и стабильный доход"
            else:
                return "Опытный профессионал с высокой квалификацией"
        except:
            return "Влияет на карьерные возможности и уровень дохода"

    if 'experience' in feature_name.lower() or 'stag' in feature_name.lower():
        try:
            exp_val = float(value)
            if exp_val < 2:
                return "Небольшой опыт — начальный этап карьеры"
            elif exp_val < 5:
                return "Средний опыт работы — развитие профессиональных навыков"
            elif exp_val < 10:
                return "Значительный опыт — хорошие карьерные перспективы"
            else:
                return "Большой опыт работы значительно повышает доходность"
        except:
            return "Опыт работы напрямую влияет на уровень дохода"

    if 'turn' in feature_name.lower() and 'cr' in feature_name.lower():
        return "Кредитовые обороты отражают активность использования кредитных средств"

    if 'turn' in feature_name.lower() and 'db' in feature_name.lower():
        return "Дебетовые обороты показывают уровень расходов и финансовую активность"

    if 'salary' in feature_name.lower():
        return "Усредненная зарплата — ключевой показатель платежеспособности"

    if 'bki' in feature_name.lower() and 'limit' in feature_name.lower():
        return "Кредитные лимиты из БКИ показывают доверие банков к клиенту"

    if 'payment' in feature_name.lower():
        return "Платежное поведение характеризует финансовую дисциплину"

    if 'by_category' in feature_name.lower():
        return "Категории транзакций отражают образ жизни и расходы клиента"

    if 'ils' in feature_name.lower():
        return "Данные из информационной системы банка о финансовой активности"

    if 'income' in feature_name.lower():
        return "Подтвержденный доход клиента по данным различных источников"

    if 'credit' in feature_name.lower() or 'cr_' in feature_name.lower():
        return "Кредитная история и активность по кредитным продуктам"

    if 'debit' in feature_name.lower() or 'db_' in feature_name.lower():
        return "Операции по дебетовым картам и счетам клиента"

    return None


def generate_importance_plot(top_features):
    """Создание графика важности признаков"""
    try:
        # Создаем фигуру
        fig, ax = plt.subplots(figsize=(12, 7))

        # Берем топ-10 признаков
        features_to_plot = top_features[:10]

        # Данные для графика
        names = [f['name'][:25] for f in features_to_plot]  # Обрезаем длинные имена
        importances = [f['normalized_importance'] for f in features_to_plot]

        # Создаем горизонтальный bar chart
        y_pos = np.arange(len(names))

        # Цветовая схема от красного к желтому
        colors = plt.cm.RdYlGn(np.linspace(0.4, 0.9, len(importances)))

        bars = ax.barh(y_pos, importances, color=colors, edgecolor='#333', linewidth=2, height=0.7)

        # Добавляем значения на графике
        for i, (bar, imp) in enumerate(zip(bars, importances)):
            width = bar.get_width()
            ax.text(width + 1.5, bar.get_y() + bar.get_height()/2,
                   f'{imp:.1f}%',
                   ha='left', va='center', fontsize=11, fontweight='bold', color='#333')

        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=11)
        ax.set_xlabel('Важность признака (%)', fontsize=13, fontweight='bold')
        ax.set_title('Топ-10 факторов, влияющих на прогноз дохода клиента',
                    fontsize=15, fontweight='bold', pad=20, color='#1A1A1A')
        ax.set_xlim(0, max(importances) * 1.2)

        # Стилизация в стиле Альфа-Банка
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666')
        ax.spines['bottom'].set_color('#666')
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=1)
        ax.set_facecolor('#FAFAFA')
        fig.patch.set_facecolor('white')

        plt.tight_layout()

        # Конвертируем в base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        return img_base64
    except Exception as e:
        print(f"Error generating importance plot: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_financial_offers(predicted_income, client_data):
    """Генерация персонализированных финансовых предложений"""
    offers = []

    # Категоризация дохода
    if predicted_income < 30000:
        income_category = "низкий"
        max_credit = predicted_income * 3
        max_card_limit = 50000
        deposit_interest = 5.5
        investment_available = False
    elif predicted_income < 80000:
        income_category = "средний"
        max_credit = predicted_income * 5
        max_card_limit = 150000
        deposit_interest = 6.0
        investment_available = False
    else:
        income_category = "высокий"
        max_credit = predicted_income * 8
        max_card_limit = 500000
        deposit_interest = 6.5
        investment_available = True

    # Предложение 1: Кредит
    offers.append({
        "type": "Потребительский кредит",
        "title": f"Кредит до {max_credit:,.0f} ₽",
        "description": f"Персональное предложение для клиентов с {income_category} уровнем дохода",
        "interest_rate": "от 9.9%",
        "term": "до 5 лет",
        "icon": "💰",
        "details": {
            "Максимальная сумма": f"{max_credit:,.0f} ₽",
            "Ежемесячный платеж": f"≈ {max_credit * 0.02:,.0f} ₽",
            "Решение": "за 1 минуту",
            "Ставка": "от 9.9% годовых"
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
            "Обслуживание": "0 ₽ при обороте от 10,000 ₽",
            "Льготный период": "100 дней"
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
    if investment_available:
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

    # Предложение 5: Дебетовая карта с кэшбэком
    offers.append({
        "type": "Дебетовая карта",
        "title": "Альфа-Карта с кэшбэком",
        "description": "До 10% кэшбэка на категории по выбору",
        "interest_rate": "бесплатное обслуживание",
        "term": "бессрочно",
        "icon": "💎",
        "details": {
            "Кэшбэк": "до 10% на категории",
            "Обслуживание": "0 ₽",
            "Снятие наличных": "без комиссии в банкоматах партнеров",
            "Бонусы": "мили за покупки"
        }
    })

    return offers


@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint для предсказания с объяснением"""
    try:
        data = request.json

        if not data:
            return jsonify({'error': 'Нет данных'}), 400

        # Получаем данные клиента
        if 'client_id' in data:
            client_id = int(data['client_id'])
            if test_data is None:
                return jsonify({'error': 'Тестовая выборка не загружена'}), 500

            if client_id not in test_data['id'].values:
                return jsonify({'error': f'Клиент с ID {client_id} не найден'}), 404

            client_row = test_data[test_data['id'] == client_id].iloc[0]
            client_data = client_row.to_dict()
        else:
            client_data = data

        # Предсказание дохода с объяснением
        predicted_income, top_features = predict_income_with_explanation(client_data)

        # Генерация графика важности признаков
        importance_plot_base64 = generate_importance_plot(top_features)

        # Генерация финансовых предложений
        offers = generate_financial_offers(predicted_income, client_data)

        # Формируем ответ с безопасной обработкой NaN
        def safe_json_value(v):
            """Безопасное преобразование значения для JSON"""
            if isinstance(v, float) and np.isnan(v):
                return None
            elif isinstance(v, (int, float)):
                return float(v)
            elif pd.isna(v):
                return None
            else:
                return str(v)

        response = {
            'predicted_income': float(predicted_income),
            'predicted_income_formatted': f"{predicted_income:,.0f} ₽",
            'top_features': top_features,
            'importance_plot': importance_plot_base64,
            'offers': offers,
            'model_type': 'Hybrid Model: FT-Transformer + CatBoost',
            'client_data': {k: safe_json_value(v) for k, v in list(client_data.items())[:10]}
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

    index_html = """<!DOCTYPE html>
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
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            color: #1A1A1A;
        }

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

        input[type="number"] {
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

        .btn-secondary {
            background: #1A1A1A;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }

        .btn-secondary:hover {
            background: #333;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
        }

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

        .visualization-section {
            background: white;
            padding: 30px;
            border-radius: 16px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }

        .visualization-section h3 {
            color: #1A1A1A;
            margin-bottom: 20px;
            font-size: 1.4em;
        }

        .importance-plot {
            max-width: 100%;
            height: auto;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
        }

        .features-list {
            display: grid;
            gap: 15px;
            margin-top: 20px;
        }

        .feature-item {
            background: white;
            padding: 20px;
            border-radius: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border: 2px solid #F0F0F0;
            transition: all 0.3s;
        }

        .feature-item:hover {
            border-color: #EF3124;
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(239, 49, 36, 0.15);
        }

        .feature-info {
            flex: 1;
        }

        .feature-name {
            font-weight: 700;
            color: #1A1A1A;
            font-size: 1.1em;
            margin-bottom: 8px;
        }

        .feature-explanation {
            color: #555;
            font-size: 0.95em;
            line-height: 1.5;
        }

        .feature-importance {
            background: linear-gradient(135deg, #EF3124 0%, #C41E3A 100%);
            color: white;
            padding: 10px 20px;
            border-radius: 25px;
            font-weight: 700;
            font-size: 1.15em;
            min-width: 80px;
            text-align: center;
            box-shadow: 0 4px 10px rgba(239, 49, 36, 0.3);
        }

        .offers-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }

        .offer-card {
            background: white;
            border-radius: 16px;
            padding: 30px;
            border: 2px solid #F0F0F0;
            transition: all 0.3s;
            cursor: pointer;
        }

        .offer-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.12);
            border-color: #EF3124;
        }

        .offer-icon {
            font-size: 3em;
            margin-bottom: 15px;
        }

        .offer-type {
            color: #EF3124;
            font-size: 0.85em;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 10px;
        }

        .offer-title {
            font-size: 1.3em;
            font-weight: 700;
            color: #1A1A1A;
            margin-bottom: 10px;
        }

        .offer-description {
            color: #666;
            margin-bottom: 20px;
            line-height: 1.6;
        }

        .offer-details {
            border-top: 1px solid #F0F0F0;
            padding-top: 20px;
        }

        .detail-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 10px;
            font-size: 0.95em;
        }

        .detail-label {
            color: #666;
        }

        .detail-value {
            font-weight: 600;
            color: #1A1A1A;
        }

        @media (max-width: 768px) {
            .input-group {
                flex-direction: column;
            }

            .logo {
                font-size: 2em;
            }

            .income-amount {
                font-size: 2.5em;
            }

            .offers-grid {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="header-content">
            <div class="logo">Альфа-Банк</div>
            <div class="tagline">AI-система прогнозирования доходов клиентов</div>
            <div class="badge">🤖 Гибридная модель FT-Transformer + CatBoost</div>
        </div>
    </div>

    <div class="container">
        <div class="card">
            <h2>🎯 Прогноз дохода клиента</h2>

            <div class="info-box">
                <div class="info-icon">💡</div>
                <div>
                    <strong>Как это работает:</strong> Введите ID клиента из базы данных, и наша AI-модель
                    спрогнозирует его доход, объяснит факторы влияния и предложит персонализированные финансовые продукты.
                </div>
            </div>

            <div class="input-group">
                <div class="form-group">
                    <label for="clientId">ID клиента</label>
                    <input type="number" id="clientId" placeholder="Например: 12345" />
                </div>
                <button class="btn" onclick="predictIncome()">
                    Прогнозировать доход
                </button>
            </div>

            <div style="margin-top: 20px;">
                <button class="btn btn-secondary" onclick="loadRandomIds()">
                    🎲 Показать случайные ID
                </button>
            </div>

            <div id="randomIds" class="random-ids"></div>
        </div>

        <div class="loader" id="loader">
            <div class="spinner"></div>
            <div class="loader-text">Анализируем данные и строим прогноз...</div>
        </div>

        <div class="result" id="result">
            <div class="card">
                <div class="income-display">
                    <h2>Прогнозируемый доход клиента</h2>
                    <div class="income-amount" id="incomeAmount">—</div>
                    <div class="model-info" id="modelInfo">Hybrid Model: FT-Transformer + CatBoost</div>
                </div>

                <div class="visualization-section" id="visualizationSection">
                    <h3>📊 Визуализация влияния факторов</h3>
                    <p style="color: #666; margin-bottom: 20px;">
                        График показывает важность каждого признака для прогнозирования дохода клиента
                    </p>
                    <img id="importancePlot" class="importance-plot" src="" alt="Feature importance visualization" />
                </div>

                <div style="margin-top: 30px;">
                    <h3 style="margin-bottom: 20px;">🎯 Ключевые факторы влияния на прогноз</h3>
                    <p style="color: #666; margin-bottom: 20px;">
                        Модель анализирует следующие характеристики клиента для определения уровня дохода:
                    </p>
                    <div class="features-list" id="featuresList"></div>
                </div>
            </div>

            <div class="card">
                <h2>💼 Персонализированные финансовые предложения</h2>
                <p style="color: #666; margin-bottom: 20px;">
                    На основе прогноза дохода мы подобрали оптимальные финансовые продукты:
                </p>
                <div class="offers-grid" id="offersGrid"></div>
            </div>
        </div>
    </div>

    <script>
        async function loadRandomIds() {
            try {
                const response = await fetch('/clients/random');
                const data = await response.json();

                const container = document.getElementById('randomIds');
                container.innerHTML = '<p style="margin-bottom: 10px; color: #666; font-weight: 600;">Кликните на ID для быстрого выбора:</p>';

                data.ids.forEach(id => {
                    const chip = document.createElement('div');
                    chip.className = 'id-chip';
                    chip.textContent = `ID: ${id}`;
                    chip.onclick = () => {
                        document.getElementById('clientId').value = id;
                        predictIncome();
                    };
                    container.appendChild(chip);
                });
            } catch (error) {
                console.error('Error loading random IDs:', error);
            }
        }

        async function predictIncome() {
            const clientId = document.getElementById('clientId').value;

            if (!clientId) {
                alert('Пожалуйста, введите ID клиента');
                return;
            }

            // Показываем loader
            document.getElementById('loader').classList.add('show');
            document.getElementById('result').classList.remove('show');

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ client_id: parseInt(clientId) })
                });

                const data = await response.json();

                if (response.ok) {
                    displayResults(data);
                } else {
                    throw new Error(data.error || 'Unknown error');
                }
            } catch (error) {
                console.error('Error:', error);
                document.getElementById('loader').classList.remove('show');
                alert('Ошибка: ' + error.message);
            }
        }

        function displayResults(data) {
            // Скрываем loader, показываем результаты
            document.getElementById('loader').classList.remove('show');
            document.getElementById('result').classList.add('show');

            // Отображаем доход
            document.getElementById('incomeAmount').textContent = data.predicted_income_formatted;
            document.getElementById('modelInfo').textContent = data.model_type;

            // Отображаем график важности признаков
            if (data.importance_plot) {
                document.getElementById('importancePlot').src = 'data:image/png;base64,' + data.importance_plot;
                document.getElementById('visualizationSection').style.display = 'block';
            } else {
                document.getElementById('visualizationSection').style.display = 'none';
            }

            // Отображаем топ-признаки
            const featuresList = document.getElementById('featuresList');
            featuresList.innerHTML = '';

            data.top_features.slice(0, 8).forEach(feature => {
                const item = document.createElement('div');
                item.className = 'feature-item';
                item.innerHTML = `
                    <div class="feature-info">
                        <div class="feature-name">${feature.name}</div>
                        <div class="feature-explanation">${feature.explanation || ''}</div>
                    </div>
                    <div class="feature-importance">
                        ${feature.normalized_importance.toFixed(1)}%
                    </div>
                `;
                featuresList.appendChild(item);
            });

            // Отображаем предложения
            const offersGrid = document.getElementById('offersGrid');
            offersGrid.innerHTML = '';

            data.offers.forEach(offer => {
                const card = document.createElement('div');
                card.className = 'offer-card';

                let detailsHtml = '';
                for (const [key, value] of Object.entries(offer.details)) {
                    detailsHtml += `
                        <div class="detail-row">
                            <span class="detail-label">${key}:</span>
                            <span class="detail-value">${value}</span>
                        </div>
                    `;
                }

                card.innerHTML = `
                    <div class="offer-icon">${offer.icon}</div>
                    <div class="offer-type">${offer.type}</div>
                    <div class="offer-title">${offer.title}</div>
                    <div class="offer-description">${offer.description}</div>
                    <div class="offer-details">
                        ${detailsHtml}
                    </div>
                `;
                offersGrid.appendChild(card);
            });

            // Плавная прокрутка к результатам
            document.getElementById('result').scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }

        // Загружаем случайные ID при загрузке страницы
        window.onload = () => {
            loadRandomIds();
        };

        // Enter для отправки
        document.getElementById('clientId').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') {
                predictIncome();
            }
        });
    </script>
</body>
</html>
"""

    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)

    print("✓ HTML шаблон создан")


if __name__ == '__main__':
    print("=" * 60)
    print("ЗАПУСК ВЕБА С AI-ПРОГНОЗИРОВАНИЕМ И ВИЗУАЛИЗАЦИЕЙ")
    print("=" * 60)

    # Создаем шаблоны
    create_templates()

    # Загружаем модель
    load_hybrid_model(
        embedder_path='hybrid_model_embedder.pth',
        catboost_path='hybrid_model.cbm'
    )

    # Загружаем тестовую выборку
    load_test_data('hackathon_income_test.csv')

    print("\n" + "=" * 60)
    print("✓ Сервер готов к работе!")
    print("=" * 60)
    print("\n📱 Откройте в браузере: http://localhost:5000")
    print("\nДоступные функции:")
    print("  ✓ Прогноз дохода клиента")
    print("  ✓ Визуализация важности признаков (Feature Importance)")
    print("  ✓ Текстовая интерпретация ('почему доход = X')")
    print("  ✓ Автоматическая генерация финансовых предложений")
    print("  ✓ Выбор клиента по ID из базы")
    print("\n" + "=" * 60 + "\n")

    app.run(debug=True, host='0.0.0.0', port=5000)
