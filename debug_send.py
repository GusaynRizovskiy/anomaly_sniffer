# -*- coding: utf-8 -*-
import json
import time
import logging
import sys
import os
import threading
import websocket  # pip install websocket-client
import ssl
from datetime import datetime, timedelta

# Включаем полную трассировку пакетов в консоль
websocket.enableTrace(True)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from remote_transmitter import RemoteTransmitter
except ImportError:
    from core.remote_transmitter import RemoteTransmitter

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("DebugSenderModule")

stop_event = threading.Event()


def heartbeat_listener(ws):
    while not stop_event.is_set():
        try:
            ws.settimeout(1.0)
            message = ws.recv()
            if message:
                data = json.loads(message)
                if data.get("type") == "administration/heartbeat":
                    ws.send(json.dumps({"type": "administration/heartbeat"}))
        except Exception:
            if not stop_event.is_set():
                stop_event.set()
            break


def run_isolated_ordered_payload_test():
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    srv = config['server']

    transmitter = RemoteTransmitter(
        base_url=f"https://{srv['ip']}:{srv['port']}",
        login=srv['login'],
        password=srv['password'],
        user_type=srv['type']
    )

    if not transmitter.authenticate():
        return

    # --- РУЧНОЕ ПОДКЛЮЧЕНИЕ С ПРЕФИКСОМ /API ---
    ws_url = f"wss://{srv['ip']}:{srv['port']}/api/integrated-container-ids/connection-integrated-container-ids?token={transmitter.token}"

    logger.info("[3] Установка WebSocket с маскировкой под браузер (User-Agent)...")
    try:
        transmitter.ws = websocket.create_connection(
            ws_url,
            sslopt={"cert_reqs": ssl.CERT_NONE},
            header=[
                "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"],
            timeout=7
        )
    except Exception as e:
        logger.error(f"[FAIL] Ошибка подключения: {e}")
        return

    # 1. Запускаем "уши" для приема Heartbeat
    threading.Thread(target=heartbeat_listener, args=(transmitter.ws,), daemon=True).start()

    try:
        # 2. Цикл отправки 5 тестовых пакетов
        # 2. Цикл отправки 5 тестовых пакетов
        for i in range(1, 6):
            if not transmitter.ws or not transmitter.ws.connected:
                logger.error(f"[FAIL] Соединение разорвано перед отправкой пакета №{i}")
                break

            # Формируем актуальное московское время (UTC+3) для каждого пакета
            moscow_time = datetime.utcnow() + timedelta(hours=3)
            ts_now = moscow_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

            raw_payload = {
                "type": "integratedContainerIds/transmittingEvents",
                "transmittingEvents": [
                    {
                        "event_type": "alert",
                        "main": {
                            "ts": ts_now,
                            "flow_id": 106703934185510 + i,  # Гарантируем уникальность id
                            "src_ip": "192.168.3.65",
                            "src_port": 63379,
                            "dest_ip": "192.168.4.66",
                            "dest_port": 443,
                            "proto": "TCP",
                            "gid": 1,
                            "signature_id": 2027695,
                            "rev": 5,
                            "signature": f"ET INFO Observed Cloudflare DNS over HTTPS Domain (Test Event №{i})",
                            "severity": 3,
                            "category": "Misc activity"
                        }
                    }
                ]
            }

            json_string = json.dumps(raw_payload, separators=(',', ':'), ensure_ascii=False)

            logger.info(f"[4] Отправка тестового пакета №{i}/5...")

            # Отправляем данные
            transmitter.ws.send(json_string)

            # ЭКСПЕРИМЕНТ №1: Увеличиваем паузу до 3-4 секунд.
            # Это гарантированно разведет пакеты по разным TCP-сегментам и обойдет Rate Limit сервера.
            time.sleep(3)

        # 3. Финальное ожидание ответов от сервера после отправки всех пакетов
        logger.info("[*] Все 5 пакетов отправлены. Ожидание 5 секунд перед закрытием сокета...")
        time.sleep(5)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if transmitter.ws:
            logger.info("[*] Закрытие WebSocket соединения...")
            transmitter.ws.close()


if __name__ == "__main__":
    run_isolated_ordered_payload_test()