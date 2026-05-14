# -*- coding: utf-8 -*-
import json
import time
import logging
import sys
import os
from datetime import datetime, timedelta

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
    logger.info("[SUCCESS] WebSocket соединен и готов к бесконечной отправке.")
    logger.info("[*] Для остановки скрипта и закрытия сокета нажмите Ctrl+C\n")

    # Переменная для подсчета отправленных пакетов (для удобства дебага)
    event_counter = 0
    # Интервал между отправками в секундах
    send_interval = 5

    try:
        while True:
            event_counter += 1

            # Динамически вычисляем московское время на момент отправки конкретного алерта
            msk_time = datetime.utcnow() + timedelta(hours=3)
            ts_msk = msk_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+03:00"

            # 5. Формируем структуру данных (структура и типы строго сохранены)
            main_body = {
                "ts": ts_msk,
                "flow_id": 1067039902185510,
                "src_ip": "200.200.220.220",
                "src_port": 3128,
                "dest_ip": "44.44.44.44",
                "dest_port": 3128,
                "proto": "TCP",
                "gid": 1,
                "signature_id": 2100366,
                "rev": 8,
                "signature": f"ET INFO Observed Cloudflare DNS over HTTPS Domain (Iteration #{event_counter})",
                "severity": 3,
                "category": "Misc activity"
            }

            # Обертка под формат напарника
            event_payload = {
                "type": "integratedContainerIds/transmittingEvents",
                "transmittingEvents": [
                    {
                        "event_type": "alert",
                        "main": main_body
                    }
                ]
            }

            json_payload_string = json.dumps(event_payload, ensure_ascii=False)

            # Вывод лога отправки
            logger.info(f"[4] Отправка события #{event_counter} в сокет... Время: {ts_msk}")

            # Проверяем, жив ли еще сокет перед отправкой
            if transmitter.ws and transmitter.ws.connected:
                transmitter.ws.send(json_payload_string)
            else:
                logger.error("[FAIL] Соединение разорвано сервером во время цикла!")
                break

            # Пауза перед следующей итерацией
            time.sleep(send_interval)

    except KeyboardInterrupt:
        # Сюда код падает при нажатии Ctrl+C
        print("\n" + "=" * 50)
        logger.info("[*] Обнаружено ручное прерывание (Ctrl+C). Корректно завершаем работу...")
        print("=" * 50)
    except Exception as e:
        logger.error(f"[FAIL] Непредвиденная ошибка в цикле отправки: {e}")
    finally:
        # 6. Закрытие соединения в любом случае (при ошибке или Ctrl+C)
        logger.info("[5] Закрытие отладочного WebSocket соединения.")
        if transmitter.ws:
            try:
                transmitter.ws.close()
                logger.info("[SUCCESS] Сокет успешно закрыт.")
            except Exception as e:
                logger.error(f"Ошибка при закрытии сокета: {e}")


if __name__ == "__main__":
    run_isolated_ordered_payload_test()