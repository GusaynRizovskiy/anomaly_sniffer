# -*- coding: utf-8 -*-
import json
import time
import logging
import sys
import os
import threading
import websocket  # pip install websocket-client
import ssl
from datetime import datetime

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

    # --- РУЧНОЕ ПОДКЛЮЧЕНИЕ В ОБХОД TRANSMITTER ДЛЯ ДОБАВЛЕНИЯ ЗАГОЛОВКОВ ---
    ws_url = f"wss://{srv['ip']}:{srv['port']}/integrated-container-ids/connection-integrated-container-ids?token={transmitter.token}"

    logger.info("[3] Установка WebSocket с маскировкой под браузер (User-Agent)...")
    try:
        transmitter.ws = websocket.create_connection(
            ws_url,
            sslopt={"cert_reqs": ssl.CERT_NONE},
            # Добавляем заголовок, который часто требуют Nginx/WAF для пропуска WS трафика
            header=[
                "User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"],
            timeout=7
        )
    except Exception as e:
        logger.error(f"[FAIL] Ошибка подключения: {e}")
        return

    # 1. Сразу запускаем "уши" для приема Heartbeat
    threading.Thread(target=heartbeat_listener, args=(transmitter.ws,), daemon=True).start()

    try:
        # 2. БЕЗ ПАУЗЫ формируем и мгновенно отправляем пакет
        ts_now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        raw_payload = {
            "type": "integratedContainerIds/transmittingEvents",
            "transmittingEvents": [
                {
                    "event_type": "alert",
                    "main": {
                        "ts": ts_now,
                        "flow_id": 1067039902185510,
                        "src_ip": "192.168.1.65",
                        "src_port": 63379,
                        "dest_ip": "192.168.1.66",
                        "dest_port": 443,
                        "proto": "TCP",
                        "gid": 1,
                        "signature_id": 2027695,
                        "rev": 5,
                        "signature": "ET INFO Observed Cloudflare DNS over HTTPS Domain (cloudflare-dns .com in TLS SNI)",
                        "severity": 3,
                        "category": "Misc activity"
                    }
                }
            ]
        }

        json_string = json.dumps(raw_payload, separators=(',', ':'), ensure_ascii=False)

        logger.info("[4] Мгновенная отправка тестового пакета...")
        if transmitter.ws and transmitter.ws.connected:
            transmitter.ws.send(json_string)

        # 3. Ждем реакцию сервера
        time.sleep(10)

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        if transmitter.ws:
            transmitter.ws.close()


if __name__ == "__main__":
    run_isolated_ordered_payload_test()