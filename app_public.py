"""
Публичная версия Flask приложения с туннелем
"""

from app import app, create_templates, load_hybrid_model, load_test_data

try:
    from flask_cloudflared import run_with_cloudflared
    USE_TUNNEL = True
except ImportError:
    print("⚠️  flask-cloudflared не установлен")
    print("   Установите: pip install flask-cloudflared")
    USE_TUNNEL = False


if __name__ == '__main__':
    print("=" * 60)
    print("ЗАПУСК ПУБЛИЧНОГО ВЕБА С AI-ПРОГНОЗИРОВАНИЕМ")
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

    if USE_TUNNEL:
        print("\n🌐 Создание публичного туннеля...")
        print("   Ждите URL для доступа...")
        print("\n" + "=" * 60 + "\n")
        run_with_cloudflared(app)
    else:
        print("\n📱 Локальный режим: http://localhost:5000")
        print("\nДля публичного доступа используйте:")
        print("  1. ssh -R 80:localhost:5000 nokey@localhost.run")
        print("  2. ssh -R 80:localhost:5000 serveo.net")
        print("\n" + "=" * 60 + "\n")
        app.run(debug=False, host='0.0.0.0', port=5000)
