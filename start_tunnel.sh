#!/bin/bash

echo "========================================"
echo "  ЗАПУСК ПУБЛИЧНОГО СЕРВЕРА"
echo "========================================"
echo ""
echo "Выберите метод:"
echo "  1) localhost.run (SSH)"
echo "  2) serveo.net (SSH)"
echo "  3) Просто локально (0.0.0.0:5000)"
echo ""
read -p "Ваш выбор (1/2/3): " choice

# Запускаем Flask в фоне
python app.py &
FLASK_PID=$!
echo "✓ Flask запущен (PID: $FLASK_PID)"
sleep 3

case $choice in
    1)
        echo ""
        echo "🌐 Создание туннеля через localhost.run..."
        echo "   (нажмите Enter если спросит про fingerprint)"
        echo ""
        ssh -o StrictHostKeyChecking=no -R 80:localhost:5000 nokey@localhost.run
        ;;
    2)
        echo ""
        echo "🌐 Создание туннеля через serveo.net..."
        echo ""
        ssh -o StrictHostKeyChecking=no -R 80:localhost:5000 serveo.net
        ;;
    3)
        IP=$(curl -s ifconfig.me 2>/dev/null || echo "YOUR_IP")
        echo ""
        echo "✓ Сервер запущен локально"
        echo ""
        echo "Доступ:"
        echo "  - Локально: http://localhost:5000"
        echo "  - По IP: http://$IP:5000"
        echo ""
        echo "Нажмите Ctrl+C для остановки"
        wait $FLASK_PID
        ;;
    *)
        echo "Неверный выбор"
        kill $FLASK_PID
        exit 1
        ;;
esac

# Cleanup
kill $FLASK_PID 2>/dev/null
