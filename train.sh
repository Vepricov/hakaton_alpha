#!/bin/bash

# Быстрый скрипт для обучения гибридной модели и вывода финального качества
# Альфа-Банк: Прогнозирование доходов
# Гибридный подход: FT-Transformer (эмбеддинги) + CatBoost (предсказание)

echo "=========================================="
echo "Обучение ГИБРИДНОЙ модели"
echo "FT-Transformer + CatBoost"
echo "=========================================="
echo ""

# Проверка виртуального окружения
if [ ! -d "venv" ]; then
    echo "Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активация venv
echo "Активация venv..."
source venv/bin/activate

# Установка зависимостей если нужно
if ! python -c "import torch" 2>/dev/null; then
    echo "Установка зависимостей..."
    pip install -q -r requirements.txt
fi

# Установка CatBoost если не установлен
if ! python -c "import catboost" 2>/dev/null; then
    echo "Установка CatBoost..."
    pip install -q catboost
fi

# Проверка наличия данных
if [ ! -f "hackathon_income_train.csv" ]; then
    echo "ОШИБКА: Файл hackathon_income_train.csv не найден!"
    exit 1
fi

if [ ! -f "hackathon_income_test.csv" ]; then
    echo "ОШИБКА: Файл hackathon_income_test.csv не найден!"
    exit 1
fi

echo ""
echo "=========================================="
echo "ЗАПУСК ОБУЧЕНИЯ ГИБРИДНОЙ МОДЕЛИ"
echo "=========================================="
echo ""
echo "Подход:"
echo "  1. FT-Transformer извлекает эмбеддинги (20 эпох с регуляризацией)"
echo "  2. CatBoost обучается на эмбеддингах + исходных признаках"
echo "  3. Решает проблему переобучения трансформера"
echo ""

# Запуск обучения гибридной модели
export CUDA_VISIBLE_DEVICES=2
python train_hybrid.py \
    --model_path hybrid_model.cbm \
    --transformer_epochs 20 \
    --batch_size 256 \
    --learning_rate 1e-4 \
    --d_token 192 \
    --n_layers 8 \
    --n_heads 8 \
    --d_ffn 200 \
    --train_path hackathon_income_train.csv \
    --test_path hackathon_income_test.csv \
    --submission_path submission_hybrid.csv \
    --generate_submission

TRAIN_EXIT_CODE=$?

echo ""
echo "=========================================="

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
    echo "ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО"
    echo "=========================================="
    echo ""

    echo "Финальное качество модели уже выведено выше"
    echo ""
    echo "=========================================="
    echo "РЕЗУЛЬТАТЫ:"
    echo "=========================================="
    echo ""

    if [ -f "hybrid_model.cbm" ]; then
        MODEL_SIZE=$(du -h hybrid_model.cbm | cut -f1)
        echo "✓ CatBoost модель: hybrid_model.cbm ($MODEL_SIZE)"
    fi

    if [ -f "hybrid_model_embedder.pth" ]; then
        EMBEDDER_SIZE=$(du -h hybrid_model_embedder.pth | cut -f1)
        echo "✓ Embedder: hybrid_model_embedder.pth ($EMBEDDER_SIZE)"
    fi

    if [ -f "submission_hybrid.csv" ]; then
        LINES=$(wc -l < submission_hybrid.csv)
        echo "✓ Submission файл: submission_hybrid.csv ($LINES строк)"
    fi

    echo ""
    echo "Преимущества гибридного подхода:"
    echo "  • Трансформер извлекает глубокие паттерны"
    echo "  • CatBoost предотвращает переобучение"
    echo "  • Лучшее качество чем у каждой модели отдельно"
    echo ""
    echo "Для запуска веб-интерфейса:"
    echo "  python app.py"
    echo ""

else
    echo "ОШИБКА ПРИ ОБУЧЕНИИ"
    echo "=========================================="
    exit 1
fi

echo "=========================================="
echo "Готово!"
echo "=========================================="
