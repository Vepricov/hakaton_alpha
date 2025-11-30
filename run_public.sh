#!/bin/bash

echo "🚀 Запуск публичного сервера..."

# Убиваем старые процессы
pkill -f "python app.py" 2>/dev/null
pkill -f "ssh.*localhost.run" 2>/dev/null
sleep 2

# Запускаем Flask
nohup python app.py > flask.log 2>&1 &
echo "✓ Flask запущен"
sleep 5

# Проверяем что Flask работает
if curl -s http://localhost:5000/health > /dev/null; then
    echo "✓ Flask отвечает"
else
    echo "✗ Flask не отвечает, проверьте flask.log"
    exit 1
fi

# Запускаем туннель с автоперезапуском
echo "🌐 Создание туннеля..."
while true; do
    ssh -o StrictHostKeyChecking=no -o ServerAliveInterval=60 -R 80:localhost:5000 nokey@localhost.run 2>&1 | tee tunnel.log
    echo "⚠️  Туннель отключился, переподключение через 5 сек..."
    sleep 5
done
