"""
Гибридная модель: FT-Transformer (эмбеддинги) + CatBoost (предсказание)
Решает проблему переобучения трансформера и улучшает качество
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from catboost import CatBoostRegressor, Pool
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from model import FTTransformer, TabularDataset
# Список признаков для удаления (из CatBoost версии)
# USELESS_FEATURES = [
#     'addrref', 'city_smart_name', 'dp_ewb_last_employment_position',
#     'client_active_flag', 'vert_has_app_ru_tinkoff_investing',
#     'dp_ewb_dismissal_due_contract_violation_by_lb_cnt', 'period_last_act_ad',
#     'ovrd_sum', 'businessTelSubs', 'dp_ils_days_ip_share_5y',
#     'nonresident_flag', 'vert_has_app_ru_vtb_invest',
#     'hdb_bki_total_pil_cnt', 'accountsalary_out_flag',
#     'id', 'dt'
# ]
#
USELESS_FEATURES = [""]


def weighted_mean_absolute_error(y_true, y_pred, weights):
    """Вычисление WMAE метрики"""
    return (weights * np.abs(y_true - y_pred)).mean()


def preprocess_data(df, is_train=True, encoders=None, scaler=None, cat_feature_names=None):
    """
    Предобработка данных

    Args:
        df: pandas DataFrame
        is_train: флаг обучающих данных
        encoders: словарь LabelEncoder'ов для категориальных признаков
        scaler: StandardScaler для числовых признаков
        cat_feature_names: список категориальных признаков

    Returns:
        X_num, X_cat, y, weights, encoders, scaler, feature_info
    """
    df_proc = df.copy()

    # Удаляем бесполезные признаки
    df_proc = df_proc.drop(columns=USELESS_FEATURES, errors='ignore')

    # Обработка смешанных типов (строки, которые на самом деле числа)
    object_cols = df_proc.select_dtypes(include='object').columns
    for col in object_cols:
        if df_proc[col].nunique() > 100:
            try:
                temp_col = df_proc[col].astype(str).str.replace(' ', '').str.replace(',', '.')
                df_proc[col] = pd.to_numeric(temp_col, errors='coerce')
            except:
                pass

    # Извлекаем таргет и веса для обучающих данных
    if is_train:
        y = df_proc['target'].values
        weights = df_proc['w'].values
        df_proc = df_proc.drop(columns=['target', 'w'], errors='ignore')
    else:
        y = None
        weights = None

    # Разделяем на числовые и категориальные признаки
    num_cols = df_proc.select_dtypes(include=['float64', 'int64']).columns.tolist()
    cat_cols = df_proc.select_dtypes(include=['object']).columns.tolist()

    # Заполняем пропуски
    df_proc[num_cols] = df_proc[num_cols].fillna(0)
    df_proc[cat_cols] = df_proc[cat_cols].fillna("MISSING")

    # Обработка числовых признаков
    X_num = df_proc[num_cols].values.astype(np.float32)

    # Нормализация числовых признаков
    if is_train:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X_num)
    else:
        if scaler is not None:
            X_num = scaler.transform(X_num)

    # Обработка категориальных признаков
    X_cat = None
    cat_cardinalities = []

    if len(cat_cols) > 0:
        if is_train:
            encoders = {}
            X_cat = np.zeros((len(df_proc), len(cat_cols)), dtype=np.int64)

            for i, col in enumerate(cat_cols):
                le = LabelEncoder()
                X_cat[:, i] = le.fit_transform(df_proc[col].astype(str))
                encoders[col] = le
                cat_cardinalities.append(len(le.classes_))
        else:
            if encoders is not None:
                X_cat = np.zeros((len(df_proc), len(cat_cols)), dtype=np.int64)
                for i, col in enumerate(cat_cols):
                    le = encoders.get(col)
                    if le is not None:
                        # Обработка неизвестных категорий
                        vals = df_proc[col].astype(str).values
                        X_cat[:, i] = np.array([
                            le.transform([v])[0] if v in le.classes_ else 0
                            for v in vals
                        ])
                        cat_cardinalities.append(len(le.classes_))

    feature_info = {
        'num_feature_names': num_cols,
        'cat_feature_names': cat_cols,
        'cat_cardinalities': cat_cardinalities,
        'n_num_features': len(num_cols),
        'n_cat_features': len(cat_cols)
    }

    return X_num, X_cat, y, weights, encoders, scaler, feature_info


class FTTransformerEmbedder(nn.Module):
    """
    FT-Transformer без regression head - только для извлечения эмбеддингов
    """
    def __init__(self, base_model):
        super().__init__()
        self.tokenizer = base_model.tokenizer
        self.cls_token = base_model.cls_token
        self.transformer_blocks = base_model.transformer_blocks

    def forward(self, x_num=None, x_cat=None):
        """
        Returns:
            embeddings: (batch_size, d_token) - эмбеддинги [CLS] токена
        """
        # Токенизация
        tokens = self.tokenizer(x_num, x_cat)

        # Добавляем CLS токен
        batch_size = tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)

        # Трансформер
        for block in self.transformer_blocks:
            tokens = block(tokens)

        # Извлекаем CLS токен как эмбеддинг
        embeddings = tokens[:, 0, :]  # (batch, d_token)

        return embeddings


def extract_embeddings(model, dataloader, device):
    """Извлечение эмбеддингов из трансформера"""
    model.eval()
    all_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings", leave=False):
            x_num = batch['x_num'].to(device) if 'x_num' in batch else None
            x_cat = batch['x_cat'].to(device) if 'x_cat' in batch else None

            embeddings = model(x_num, x_cat)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.concatenate(all_embeddings, axis=0)


def train_transformer_embedder(args):
    """
    Этап 1: Обучение FT-Transformer для получения хороших эмбеддингов
    (с меньшим количеством эпох и большей регуляризацией)
    """
    print("\n" + "="*80)
    print("ЭТАП 1: ОБУЧЕНИЕ FT-TRANSFORMER ДЛЯ ЭМБЕДДИНГОВ")
    print("="*80)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"\nУстройство: {device}")

    # Загрузка данных
    print("\n1. Загрузка данных...")
    train_df = pd.read_csv(args.train_path, decimal=',', sep=';', low_memory=False)
    test_df = pd.read_csv(args.test_path, decimal=',', sep=';', low_memory=False)
    print(f"   Train: {train_df.shape}")
    print(f"   Test: {test_df.shape}")

    # Предобработка
    print("\n2. Предобработка...")
    X_num_full, X_cat_full, y_full, weights_full, encoders, scaler, feature_info = preprocess_data(
        train_df, is_train=True
    )

    y_log_full = np.log1p(y_full)

    # Train/Val split
    X_num_train, X_num_val, X_cat_train, X_cat_val, y_train, y_val, w_train, w_val = train_test_split(
        X_num_full, X_cat_full, y_log_full, weights_full, test_size=0.2, random_state=args.seed
    )

    print(f"   Train размер: {len(X_num_train)}")
    print(f"   Val размер: {len(X_num_val)}")

    # Датасеты
    train_dataset = TabularDataset(X_num_train, X_cat_train, y_train, w_train)
    val_dataset = TabularDataset(X_num_val, X_cat_val, y_val, w_val)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # Модель (с большей регуляризацией для эмбеддингов)
    print("\n3. Создание FT-Transformer...")
    from model import FTTransformer, WeightedMAELoss

    # Сохраняем конфигурацию модели
    model_config = {
        'n_num_features': feature_info['n_num_features'],
        'cat_cardinalities': feature_info['cat_cardinalities'] if feature_info['n_cat_features'] > 0 else [],
        'd_token': args.d_token,
        'n_layers': args.n_layers,
        'n_heads': args.n_heads,
        'd_ffn': args.d_ffn,
        'dropout': 0.2,
        'attention_dropout': 0.3
    }

    model = FTTransformer(**model_config).to(device)

    print(f"   Параметров: {sum(p.numel() for p in model.parameters()):,}")

    # Обучение (меньше эпох, только для хороших эмбеддингов)
    criterion = WeightedMAELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)

    print(f"\n4. Обучение трансформера ({args.transformer_epochs} эпох)...")

    best_loss = float('inf')
    patience_counter = 0

    for epoch in range(args.transformer_epochs):
        # Train
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
            x_num = batch['x_num'].to(device) if 'x_num' in batch else None
            x_cat = batch['x_cat'].to(device) if 'x_cat' in batch else None
            y = batch['y'].to(device)
            w = batch['weights'].to(device)

            optimizer.zero_grad()
            predictions = model(x_num, x_cat)
            loss = criterion(predictions, y, w)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                x_num = batch['x_num'].to(device) if 'x_num' in batch else None
                x_cat = batch['x_cat'].to(device) if 'x_cat' in batch else None
                y = batch['y'].to(device)
                w = batch['weights'].to(device)

                predictions = model(x_num, x_cat)
                loss = criterion(predictions, y, w)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"  Эпоха {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'transformer_embedder_temp.pth')
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print(f"  Early stopping на эпохе {epoch+1}")
                break

    # Загружаем лучшую модель
    model.load_state_dict(torch.load('transformer_embedder_temp.pth'))

    # Создаем embedder (без regression head)
    embedder = FTTransformerEmbedder(model)
    embedder.eval()

    return embedder, device, feature_info, encoders, scaler, X_num_full, X_cat_full, y_full, weights_full, model_config


def train_catboost_on_embeddings(embedder, device, feature_info, X_num, X_cat, y, weights, args):
    """
    Этап 2: Обучение CatBoost на эмбеддингах трансформера
    """
    print("\n" + "="*80)
    print("ЭТАП 2: ОБУЧЕНИЕ CATBOOST НА ЭМБЕДДИНГАХ")
    print("="*80)

    # Извлечение эмбеддингов для всего train
    print("\n1. Извлечение эмбеддингов...")
    full_dataset = TabularDataset(X_num, X_cat, None, None)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    embeddings = extract_embeddings(embedder, full_loader, device)
    print(f"   Эмбеддинги: {embeddings.shape}")

    # Комбинируем эмбеддинги с исходными признаками
    print("\n2. Комбинирование признаков...")
    # Добавляем эмбеддинги как дополнительные числовые признаки
    X_combined = np.concatenate([X_num, embeddings], axis=1)
    print(f"   Исходные числовые: {X_num.shape[1]}")
    print(f"   Эмбеддинги: {embeddings.shape[1]}")
    print(f"   Итого признаков: {X_combined.shape[1]}")

    # Логарифмируем таргет для CatBoost
    y_log = np.log1p(y)

    # Train/Val split
    X_train, X_val, y_train, y_val, w_train, w_val = train_test_split(
        X_combined, y_log, weights, test_size=0.2, random_state=args.seed
    )

    # CatBoost
    print("\n3. Обучение CatBoost...")

    train_pool = Pool(X_train, y_train, weight=w_train)
    val_pool = Pool(X_val, y_val, weight=w_val)

    model = CatBoostRegressor(
        iterations=5000,
        learning_rate=0.03,
        depth=10,
        l2_leaf_reg=3,
        loss_function='RMSE',
        eval_metric='MAE',
        random_seed=args.seed,
        verbose=100,
        early_stopping_rounds=100,
        task_type='GPU' if torch.cuda.is_available() and not args.cpu else 'CPU'
    )

    model.fit(train_pool, eval_set=val_pool)

    # Валидация
    print("\n4. Финальная валидация...")
    predictions_log = model.predict(X_val)

    # Отладочная информация
    print(f"\n   Debug info:")
    print(f"   predictions_log: min={predictions_log.min():.4f}, max={predictions_log.max():.4f}, mean={predictions_log.mean():.4f}")
    print(f"   NaN в predictions_log: {np.any(np.isnan(predictions_log))}")

    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    print(f"   predictions: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
    print(f"   NaN в predictions: {np.any(np.isnan(predictions))}")

    # y_val уже в логарифмическом виде, преобразуем обратно
    y_val_real = np.expm1(y_val)

    print(f"   y_val_real: min={y_val_real.min():.2f}, max={y_val_real.max():.2f}, mean={y_val_real.mean():.2f}")
    print(f"   w_val: min={w_val.min():.4f}, max={w_val.max():.4f}, mean={w_val.mean():.4f}")
    print(f"   NaN в y_val_real: {np.any(np.isnan(y_val_real))}")
    print(f"   NaN в w_val: {np.any(np.isnan(w_val))}")

    # Проверка на NaN
    if np.any(np.isnan(predictions)):
        print("\n   WARNING: NaN в предсказаниях! Заменяем на среднее значение.")
        predictions = np.nan_to_num(predictions, nan=y_val_real.mean())

    if np.any(np.isnan(y_val_real)):
        print("\n   WARNING: NaN в y_val_real!")
        y_val_real = np.nan_to_num(y_val_real, nan=0)

    if np.any(np.isnan(w_val)):
        print("\n   WARNING: NaN в весах!")
        w_val = np.nan_to_num(w_val, nan=1.0)

    wmae = weighted_mean_absolute_error(y_val_real, predictions, w_val)

    print(f"\n   Вычисленный WMAE: {wmae:.2f}")
    print(f"   WMAE is NaN: {np.isnan(wmae)}")

    print("\n" + "="*80)
    print(f"ФИНАЛЬНЫЙ VALIDATION WMAE: {wmae:.2f}")
    print("="*80)

    return model, embeddings.shape[1]


def generate_submission(embedder, catboost_model, device, feature_info, encoders, scaler,
                       test_path, output_path, args):
    """
    Этап 3: Генерация submission
    """
    print("\n" + "="*80)
    print("ЭТАП 3: ГЕНЕРАЦИЯ SUBMISSION")
    print("="*80)

    # Загрузка test
    print("\n1. Загрузка test данных...")
    test_df = pd.read_csv(test_path, decimal=',', sep=';', low_memory=False)

    # Предобработка
    print("\n2. Предобработка...")
    X_num_test, X_cat_test, _, _, _, _, _ = preprocess_data(
        test_df, is_train=False, encoders=encoders, scaler=scaler,
        cat_feature_names=feature_info['cat_feature_names']
    )

    # Извлечение эмбеддингов
    print("\n3. Извлечение эмбеддингов...")
    test_dataset = TabularDataset(X_num_test, X_cat_test, None, None)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    test_embeddings = extract_embeddings(embedder, test_loader, device)

    # Комбинирование
    X_test_combined = np.concatenate([X_num_test, test_embeddings], axis=1)

    # Предсказание
    print("\n4. Предсказание...")
    predictions_log = catboost_model.predict(X_test_combined)
    predictions = np.expm1(predictions_log)
    predictions = np.maximum(predictions, 0)

    # Submission
    print("\n5. Создание submission...")
    submission = test_df[['id']].copy()
    submission['target'] = predictions
    submission.to_csv(output_path, index=False)

    print(f"\n   ✓ Submission сохранен: {output_path}")
    print(f"   Средний доход: {predictions.mean():,.2f} ₽")
    print(f"   Медианный доход: {np.median(predictions):,.2f} ₽")


def save_hybrid_model(embedder, catboost_model, feature_info, encoders, scaler,
                      n_embeddings, output_path, model_config=None):
    """Сохранение гибридной модели"""
    print("\n6. Сохранение модели...")

    # Сохраняем embedder
    torch.save({
        'embedder_state_dict': embedder.state_dict(),
        'feature_info': feature_info,
        'encoders': encoders,
        'scaler': scaler,
        'n_embeddings': n_embeddings,
        'model_config': model_config
    }, output_path.replace('.cbm', '_embedder.pth'))

    # Сохраняем конфигурацию модели в JSON для легкой загрузки
    if model_config:
        config_path = output_path.replace('.cbm', '_config.json')
        with open(config_path, 'w') as f:
            json.dump(model_config, f, indent=2)
        print(f"   ✓ Config: {config_path}")

    # Сохраняем CatBoost
    catboost_model.save_model(output_path)

    print(f"   ✓ Embedder: {output_path.replace('.cbm', '_embedder.pth')}")
    print(f"   ✓ CatBoost: {output_path}")


def main(args):
    """Главная функция"""

    print("\n" + "="*80)
    print("ГИБРИДНАЯ МОДЕЛЬ: FT-Transformer + CatBoost")
    print("="*80)
    print("\nПодход:")
    print("  1. FT-Transformer извлекает глубокие эмбеддинги из данных")
    print("  2. CatBoost строит финальное предсказание на эмбеддингах + исходных признаках")
    print("  3. Решает проблему переобучения трансформера")
    print("="*80)

    # Этап 1: Обучение трансформера для эмбеддингов
    embedder, device, feature_info, encoders, scaler, X_num, X_cat, y, weights, model_config = \
        train_transformer_embedder(args)

    # Этап 2: Обучение CatBoost на эмбеддингах
    catboost_model, n_embeddings = train_catboost_on_embeddings(
        embedder, device, feature_info, X_num, X_cat, y, weights, args
    )

    # Этап 3: Генерация submission
    if args.generate_submission:
        generate_submission(
            embedder, catboost_model, device, feature_info, encoders, scaler,
            args.test_path, args.submission_path, args
        )

    # Сохранение модели
    save_hybrid_model(
        embedder, catboost_model, feature_info, encoders, scaler,
        n_embeddings, args.model_path, model_config
    )

    print("\n" + "="*80)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
    print("="*80)
    print("\nДля использования модели:")
    print("  python demo_hybrid.py validate")
    print("  python app_hybrid.py")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Hybrid Model: FT-Transformer + CatBoost')

    # Пути
    parser.add_argument('--train_path', type=str, default='hackathon_income_train.csv')
    parser.add_argument('--test_path', type=str, default='hackathon_income_test.csv')
    parser.add_argument('--model_path', type=str, default='hybrid_model.cbm')
    parser.add_argument('--submission_path', type=str, default='submission_hybrid.csv')

    # Параметры трансформера
    parser.add_argument('--d_token', type=int, default=192)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--d_ffn', type=int, default=512)

    # Параметры обучения
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--transformer_epochs', type=int, default=20,
                        help='Epochs for transformer (меньше чем обычно)')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=42)

    # Флаги
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--generate_submission', action='store_true', default=True)

    args = parser.parse_args()

    main(args)
