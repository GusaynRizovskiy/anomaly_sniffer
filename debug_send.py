# -*- coding: utf-8 -*-
import json
import time
import logging
import sys
import os
from datetime import datetime

# Добавляем путь к core, если файлы лежат в разных папках
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from remote_transmitter import RemoteTransmitter
except ImportError:
    from core.remote_transmitter import RemoteTransmitter

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DebugSenderModule")


def run_isolated_ordered_payload_test():
    # 1. Читаем конфигурацию
    try:
        with open('config.json', 'r', encoding='utf-8') as f:
            config = json.load(f)
        srv = config['server']
        logger.info(f"[1] Конфиг успешно считан. Сервер: {srv['ip']}:{srv['port']}")
    except Exception as e:
        logger.error(f"[FAIL] Ошибка чтения config.json: {e}")
        return

    # 2. Инициализируем передатчик
    base_url = f"https://{srv['ip']}:{srv['port']}"
    transmitter = RemoteTransmitter(
        base_url=base_url,
        login=srv['login'],
        password=srv['password'],
        user_type=srv['type']
    )

    # 3. Аутентификация
    logger.info("[2] Запуск аутентификации...")
    if not transmitter.authenticate():
        logger.error("[FAIL] Авторизация не удалась.")
        return
    logger.info("[SUCCESS] Авторизация успешно пройдена.")

    # 4. WebSocket соединение
    logger.info("[3] Установка WebSocket соединения...")
    transmitter.connect_ws()

    if not transmitter.ws or not transmitter.ws.connected:
        logger.error("[FAIL] WebSocket соединение не установлено.")
        return
    logger.info("[SUCCESS] WebSocket соединен и готов к отправке.")

    # 5. Формируем структуру данных в требуемом ПОРЯДКЕ полей
    # 5. Формируем структуру данных с учетом строгих типов сервера (строки вместо int)
    # Используем существующий в базе sid для проверки прохождения фильтрации
    m_data = {
        "m.gid": "1",  # Сервер ждет СТРОКУ
        "m.signature_id": 2100366,  # Берем реальный существующий SID из лога (строка!)
        "m.rev": "8",  # Тоже строка
        "m.category": "Misc activity",  # Реальная категория
        "m.signature": "23532",  # Поменяем хвост для дебага
        "m.ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
        "m.flow_id": "5555",  # Сервер ждет строку!
        "m.severity": 3,  # Severity у нас мапился в int (INFO = 1)
        "m.proto": "RTP",  # Проверенный протокол
        "m.src_ip": "200.200.220.220",
        "m.src_port": "3128",  # Для ICMP сервер вернул null, передаем None
        "m.dest_ip": "44.44.44.44",
        "m.dest_port": "3128"  # Для ICMP передаем None
    }

    event_payload = {"main": m_data}

    # В Python 3.7+ обычные словари (dict) по дефолту сохраняют порядок вставки ключей,
    # поэтому json.dumps() соберет строку в той же последовательности.
    json_payload_string = json.dumps(event_payload)

    # Выводим в консоль для проверки структуры и порядка
    print("\n" + "=" * 50)
    print("[DEBUG] СФОРМИРОВАННЫЙ ПАКЕТ В СТРОГОМ ПОРЯДКЕ:")
    print(json.dumps(event_payload, indent=2, ensure_ascii=False))
    print("=" * 50 + "\n")

    # 6. Отправка пакета в сокет
    try:
        logger.info("[4] Отправка упорядоченного пакета в WebSocket...")
        transmitter.ws.send(json_payload_string)
        logger.info("[SUCCESS] Упорядоченные данные успешно вытолкнуты в сокет.")

        # Пауза перед завершением
        time.sleep(2)

    except Exception as e:
        logger.error(f"[FAIL] Ошибка непосредственно при отправке данных: {e}")
    finally:
        logger.info("[5] Закрытие отладочного соединения.")
        transmitter.ws.close()


if __name__ == "__main__":
    run_isolated_ordered_payload_test()