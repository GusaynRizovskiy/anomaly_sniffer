# -*- coding: utf-8 -*-
import json
import websocket
import logging
import ssl
import requests
import urllib3
import threading
import time
from queue import Queue, Empty
from datetime import datetime, timedelta

# Отключаем предупреждения SSL, так как работаем по IP
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)


class RemoteTransmitter:
    def __init__(self, base_url, login, password, user_type):
        self.raw_url = base_url.replace("https://", "").replace("http://", "")
        self.base_url = base_url
        self.login = login
        self.password = password
        self.user_type = user_type
        self.token = None
        self.ws = None

        # Потокобезопасная очередь для плавной отправки событий
        self.event_queue = Queue()
        self.running = False

        # Потоковые объекты
        self.worker_thread = None
        self.heartbeat_thread = None
        self.lock = threading.Lock()  # Для безопасного доступа к self.ws из разных потоков

    def _get_severity_local(self, mse, threshold):
        ratio = mse / (threshold if threshold > 0 else 1)
        if ratio > 3.0: return "CRITICAL"
        if ratio > 1.5: return "WARNING"
        return "INFO"
    def get_attack_history(self, minutes_back=30):
        if not self.token:
            print("[CRITICAL] Нет токена! Проверьте авторизацию.")
            return []

        api_url = f"{self.base_url}/api/charts/get-attack-retrospective"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # Генерируем корректные временные метки (как в успешном ТЕСТЕ 1)
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=int(minutes_back))

        def format_ts_simple(dt):
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

        payload = {
            "filters": {
                "unitType": "minute",
                "unitValue": int(minutes_back),  # Передаем строго как int
                "tsEnd": format_ts_simple(now),
                "tsStart": format_ts_simple(start_time),
                "groupSensorsIDSelected": []
            },
            "params": {
                "sortBy": "id",
                "sortDir": "ASC",
                "page": 1,
                "pageSize": 20,
                "search": ""
            }
        }

        try:
            response = requests.post(api_url, json=payload, headers=headers, verify=False, timeout=15)

            if response.status_code != 200:
                print(f"[API ERROR] Статус: {response.status_code}, Ответ: {response.text}")
                return []

            raw_json = response.json()

            if isinstance(raw_json, dict):
                data = raw_json.get('rows')

                if data is None:
                    data = raw_json.get('items') or raw_json.get('data') or raw_json.get('content')

                if data is None:
                    print(f"[API DEBUG] Не удалось найти список событий. Ключи в ответе: {list(raw_json.keys())}")
                    return []

                print(f"[API DEBUG] Получено записей из 'rows': {len(data)}")

                # ================= ВРЕМЕННЫЙ ДЕБАГ КЛЮЧЕЙ =================
                if len(data) > 0:
                    print("\n[API DEBUG] === СТРУКТУРА ОДНОГО СОБЫТИЯ ОТ СЕРВЕРА ===")
                    print(json.dumps(data[0], indent=4, ensure_ascii=False))
                    print("====================================================\n")
                # ==========================================================

                return data

            elif isinstance(raw_json, list):
                return raw_json

            return []

        except Exception as e:
            print(f"[API ERROR] Ошибка при запросе истории: {e}")
            return []

    def authenticate(self):
        """Получение accessToken через REST API."""
        url = f"{self.base_url}/api/auth/login"
        payload = {"login": self.login, "password": self.password, "type": self.user_type}
        headers = {"Content-Type": "application/json"}

        try:
            print(f"\n[*] Аутентификация на сервере: {url}")
            response = requests.post(url, json=payload, headers=headers, verify=False, timeout=10)

            if not response.ok:
                print(f"[FAIL] Ошибка авторизации: {response.status_code}, {response.text}")
                return False

            data = response.json()
            self.token = data.get('accessToken')
            print("[SUCCESS] Токен успешно получен.")
            return True
        except Exception as e:
            print(f"[ERROR] Ошибка при выполнении аутентификации: {e}")
            return False

    def connect_ws(self):
        """Установка стабильного WebSocket соединения и запуск фоновых процессов."""
        with self.lock:
            if self.ws and self.ws.connected:
                return True

            if not self.token and not self.authenticate():
                print("[FAIL] Нет токена для WebSocket сессии.")
                return False

            ws_url = f"wss://{self.raw_url}/api/integrated-container-ids/connection-integrated-container-ids?token={self.token}"
            print(f"[*] Открытие постоянного канала WebSocket... \n[*] URL: {ws_url}")

            try:
                self.ws = websocket.create_connection(
                    ws_url,
                    sslopt={"cert_reqs": ssl.CERT_NONE},
                    timeout=7
                )
                print("[SUCCESS] Постоянный канал WebSocket успешно активирован!")

                # Запускаем фоновые потоки, если они еще не запущены
                if not self.running:
                    self.running = True

                    # 1. Поток удержания сессии (Heartbeat/Пинги)
                    self.heartbeat_thread = threading.Thread(target=self._heartbeat_listener, daemon=True)
                    self.heartbeat_thread.start()

                    # 2. Поток плавной отправки из очереди
                    self.worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
                    self.worker_thread.start()

                return True
            except Exception as e:
                print(f"[ERROR] Не удалось установить WebSocket ({type(e).__name__}): {e}")
                self.ws = None
                return False

    def _heartbeat_listener(self):
        """Фоновый поток: отвечает на пинги сервера, предотвращая закрытие сокета по таймауту."""
        while self.running:
            try:
                # Безопасно забираем ссылку на сокет
                with self.lock:
                    current_ws = self.ws
                    if not current_ws or not current_ws.connected:
                        current_ws = None

                if current_ws:
                    current_ws.settimeout(1.0)
                    try:
                        message = current_ws.recv()
                        if message:
                            data = json.loads(message)
                            # Если сервер прислал хартбит, мгновенно шлем ответ
                            if data.get("type") == "administration/heartbeat":
                                with self.lock:
                                    if self.ws and self.ws.connected:
                                        self.ws.send(json.dumps({"type": "administration/heartbeat"}))
                                        logger.debug("Отправлен ответ на Heartbeat сервера.")
                    except websocket.WebSocketTimeoutException:
                        continue  # Обычный таймаут ожидания данных, идем на следующий цикл
                else:
                    time.sleep(1)
            except Exception as e:
                logger.debug(f"Исключение в потоке Heartbeat: {e}")
                time.sleep(2)

    def _queue_worker(self):
        """Фоновый поток: забирает пакеты из очереди и шлет их строго с задержкой в 3 секунды."""
        while self.running:
            try:
                # Ждем появления аномалии в очереди (таймаут, чтобы поток проверял флаг self.running)
                internal_anomaly_data = self.event_queue.get(timeout=1)
            except Empty:
                continue

            # Пытаемся отправить, пока пакет не уйдет (с автопереподключением)
            sent = False
            while not sent and self.running:
                # Проверяем сокет, если упал — поднимаем
                if not self.ws or not self.ws.connected:
                    logger.warning("Обнаружен разрыв канала. Попытка восстановить соединение...")
                    # Переполучаем токен на случай, если старый протух
                    self.authenticate()
                    if not self.connect_ws():
                        logger.error("Реконнект не удался. Повтор через 5 секунд...")
                        time.sleep(5)
                        continue

                try:
                    # Формируем payload (актуальное московское время на момент РЕАЛЬНОЙ отправки)
                    payload_string = self._prepare_payload_string(internal_anomaly_data)

                    with self.lock:
                        if self.ws and self.ws.connected:
                            self.ws.send(payload_string)
                            sent = True
                            print(f"[NETWORK SUCCESS] Аномалия плавно отправлена из очереди на сервер.")
                            logger.info("Событие аномалии успешно доставлено.")
                except Exception as e:
                    logger.error(f"Ошибка при физической отправке фрейма: {e}. Повторный реконнект...")
                    with self.lock:
                        if self.ws:
                            try:
                                self.ws.close()
                            except:
                                pass
                            self.ws = None
                    time.sleep(2)

            self.event_queue.task_done()

            # ВАЖНО: Выдерживаем паузу в 3 секунды МЕЖДУ отправками, чтобы сервер гарантированно обработал лог!
            if sent:
                time.sleep(3)

    def _prepare_payload_string(self, internal_anomaly_data):
        """Вспомогательный метод сборки правильной JSON-строки"""
        ctx = internal_anomaly_data.get('network_context', {})
        defaults = {'src_ip': '0.0.0.0', 'dst_ip': '0.0.0.0', 'src_port': 0, 'dst_port': 0, 'protocol': 'UNKNOWN'}

        def validate(val, key):
            if val is None or val == "" or val == 0 or val == "0.0.0.0":
                return defaults.get(key, "N/A")
            return val

        severity_map = {"CRITICAL": 1, "WARNING": 2, "INFO": 3}
        local_sev = self._get_severity_local(internal_anomaly_data.get('mse_error', 0),
                                             internal_anomaly_data.get('threshold', 1))
        numeric_severity = severity_map.get(local_sev, 3)

        # Московское время (UTC+3)
        moscow_time = datetime.utcnow() + timedelta(hours=3)
        ts_now = moscow_time.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        main_data = {
            "ts": ts_now,
            "flow_id": int(internal_anomaly_data.get('flow_id', 1067039902185510)),
            "src_ip": str(validate(ctx.get('src_ip'), 'src_ip')),
            "src_port": int(validate(ctx.get('src_port'), 'src_port')),
            "dest_ip": str(validate(ctx.get('dst_ip'), 'dst_ip')),
            "dest_port": int(validate(ctx.get('dst_port'), 'dst_port')),
            "proto": str(validate(ctx.get('protocol'), 'protocol')).upper(),
            "gid": 1,
            "signature_id": int(internal_anomaly_data.get('signature_id', 2027695)),
            "rev": 1,
            "signature": f"Anomaly Detected ({internal_anomaly_data.get('anomaly_score', 0):.2f}%)",
            "severity": numeric_severity,
            "category": "Network Anomaly"
        }

        final_payload = {
            "type": "integratedContainerIds/transmittingEvents",
            "transmittingEvents": [{"event_type": "alert", "main": main_data}]
        }
        return json.dumps(final_payload, separators=(',', ':'), ensure_ascii=False)

    def send_event(self, internal_anomaly_data):
        """
        Основной метод, вызываемый из main.py.
        Теперь он мгновенно закидывает событие в очередь и НЕ блокирует сниффер.
        """
        # Если вдруг очереди нет или воркер остановлен
        if not self.running:
            logger.error("Передатчик не запущен или остановлен. Отправка невозможна.")
            return False

        self.event_queue.put(internal_anomaly_data)
        logger.info(f"Событие поставлено в очередь на отправку. Текущий размер очереди: {self.event_queue.qsize()}")
        return True

    def logout(self):
        """Завершение работы передатчика и очистка ресурсов."""
        self.running = False
        print("[*] Остановка фоновых потоков и закрытие сессии...")

        if self.token:
            try:
                url = f"{self.base_url}/api/auth/logout"
                headers = {"Authorization": f"Bearer {self.token}"}
                requests.get(url, headers=headers, verify=False, timeout=3)
            except:
                pass
            self.token = None

        with self.lock:
            if self.ws:
                try:
                    self.ws.close()
                except:
                    pass
                self.ws = None