# -*- coding: utf-8 -*-
import json
import websocket
import logging
import ssl
import requests
import urllib3
from datetime import datetime, timedelta
# Отключаем предупреждения SSL, так как работаем по IP
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Настраиваем локальный логгер для этого модуля, чтобы избежать циклического импорта
logger = logging.getLogger(__name__)

class RemoteTransmitter:
    def __init__(self, base_url, login, password, user_type):
        # Удаляем https:// для формирования ws url
        self.raw_url = base_url.replace("https://", "").replace("http://", "")
        self.base_url = base_url
        self.login = login
        self.password = password
        self.user_type = user_type  # Сохраняем тип пользователя
        self.token = None
        self.ws = None

    def _get_severity_local(self, mse, threshold):
        """Дублируем логику, чтобы не зависеть от main.py"""
        ratio = mse / threshold
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

    def _parse_diagnostic_response(self, raw_json):
        """Вспомогательный метод разбора ответа"""
        if isinstance(raw_json, dict):
            # Проверяем ключи, в которых SIEM обычно возвращает массивы
            data = raw_json.get('items') or raw_json.get('data') or raw_json.get('content') or raw_json.get('results')
            if data is not None:
                return data
            # Если ключей нет, проверим структуру
            print(f"[API DEBUG] Нетипичная структура словаря. Ключи: {list(raw_json.keys())}")
        elif isinstance(raw_json, list):
            return raw_json
        return None


    def authenticate(self):
        """Получение accessToken через REST API с детальной обработкой ошибок."""
        url = f"{self.base_url}/api/auth/login"
        payload = {
            "login": self.login,
            "password": self.password,
            "type": self.user_type
        }

        # Явно указываем заголовок, чтобы сервер понимал, что мы шлем JSON
        headers = {
            "Content-Type": "application/json"
        }

        try:
            # Выводим отладочную информацию в консоль перед отправкой
            print(f"\n[*] Отправка запроса на: {url}")
            print(f"[*] Тело запроса (payload): {json.dumps(payload, ensure_ascii=False)}")

            # verify=False для работы с самоподписанными сертификатами
            response = requests.post(url, json=payload, headers=headers, verify=False, timeout=10)

            # Если сервер вернул ошибку (4xx или 5xx код)
            if not response.ok:
                print(f"\n[FAIL] Сервер вернул код ошибки: {response.status_code}")
                # Выводим точный ответ сервера, где обычно написана причина ошибки 400
                print(f"[FAIL] Ответ сервера (детали): {response.text}")

                # Вызываем исключение HTTPError для перехода в соответствующий блок except
                response.raise_for_status()

            # Если все успешно (код 200)
            data = response.json()
            self.token = data.get('accessToken')
            print("[SUCCESS] Токен успешно получен.")
            return True

        except requests.exceptions.ConnectionError:
            print(f"[ERROR] Ошибка сети: Сервер {self.base_url} недоступен. Проверьте IP/Порт или Firewall.")
        except requests.exceptions.Timeout:
            print("[ERROR] Превышено время ожидания ответа от сервера (Timeout).")
        except requests.exceptions.HTTPError as e:
            # Проверяем код состояния через e.response
            if e.response is not None:
                if e.response.status_code == 401:
                    print("[ERROR] Ошибка 401: Неверный логин или пароль.")
                elif e.response.status_code == 400:
                    print("[ERROR] Ошибка 400: Некорректный запрос (Bad Request). Изучите детали ответа сервера выше.")
                else:
                    print(f"[ERROR] Ошибка HTTP: {e.response.status_code}")
            else:
                print(f"[ERROR] Ошибка HTTP: {e}")
        except Exception as e:
            print(f"[ERROR] Непредвиденная ошибка: {e}")

        return False

    def logout(self):
        """Завершение сессии на сервере."""
        if not self.token:
            return

        try:
            url = f"{self.base_url}/api/auth/logout"
            headers = {"Authorization": f"Bearer {self.token}"}
            # Обычно logout — это GET или POST запрос
            response = requests.get(url, headers=headers, verify=False, timeout=5)

            if response.status_code == 200:
                logger.info("Сессия успешно завершена (Logout).")
                self.token = None
                if self.ws:
                    self.ws.close()
            else:
                logger.error(f"Не удалось корректно выйти: {response.status_code}")
        except Exception as e:
            logger.error(f"Ошибка при попытке выхода: {e}")

    def connect_ws(self):
        """Установка WebSocket соединения с детальной диагностикой."""
        if not self.token:
            if not self.authenticate():
                print("[FAIL] Не удалось получить токен для WebSocket.")
                return

        # Формируем URL динамически на основе base_url
        ws_url = f"wss://{self.raw_url}/integrated-container-ids/connection-integrated-container-ids?token={self.token}"

        print(f"\n[*] Попытка установить WebSocket соединение...")
        print(f"[*] URL: {ws_url}")

        try:
            # Используем стандартный ssl.CERT_NONE вместо несуществующего websocket.ssl
            self.ws = websocket.create_connection(
                ws_url,
                sslopt={"cert_reqs": ssl.CERT_NONE},  # <- ИСПРАВЛЕНО ТУТ
                timeout=7
            )
            print("[SUCCESS] WebSocket соединение успешно установлено!")
            logger.info("WebSocket соединение установлено.")
            return True

        except websocket.WebSocketConnectionClosedException:
            print("[ERROR] Сервер принудительно закрыл WebSocket соединение сразу после подключения.")
        except websocket.WebSocketTimeoutException:
            print("[ERROR] Превышено время ожидания (Timeout) при попытке установить WebSocket соединение.")
        except ConnectionRefusedError:
            print(
                f"[ERROR] Соединение отклонено сервером. Проверьте, запущен ли WebSocket-сервер на порту {self.raw_url.split(':')[-1]}")
        except Exception as e:
            # Выводим тип ошибки и её описание для точечной диагностики
            print(f"[ERROR] Непредвиденная ошибка WebSocket ({type(e).__name__}): {e}")
            logger.error(f"Ошибка WebSocket: {e}")
            self.ws = None

    def send_event(self, internal_anomaly_data):
        """
        Отправка события через WebSocket.
        Удаляет пустые поля и использует префикс 'm.' для ключей.
        """
        if not self.token: return False

        try:
            if not self.ws or not hasattr(self.ws, 'connected') or not self.ws.connected:
                self.connect_ws()

            if self.ws and self.ws.connected:
                severity_map = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
                ctx = internal_anomaly_data.get('network_context', {})

                # Собираем данные
                m_data = {
                    "m.signature_id": 1,
                    "m.category": "Network Anomaly",
                    "m.signature": f"Anomaly Detected ({internal_anomaly_data.get('anomaly_score', 0)}%)",
                    "m.ts": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z",
                    "m.severity": severity_map.get(self._get_severity_local(internal_anomaly_data['mse_error'],
                                                                            internal_anomaly_data['threshold']), 1),
                    "m.proto": ctx.get('protocol', 'TCP').upper(),
                    "m.src_ip": ctx.get('src_ip', '0.0.0.0'),
                    "m.src_port": int(ctx.get('src_port', 0)),
                    "m.dest_ip": ctx.get('dst_ip', '0.0.0.0'),
                    "m.dest_port": int(ctx.get('dst_port', 0))
                }

                # ФИЛЬТРАЦИЯ: Удаляем ключи, где значение None или пустая строка
                # (Согласно требованию: "поля, которые не можешь заполнить, просто не передавай")
                m_filtered = {k: v for k, v in m_data.items() if v is not None and v != ""}

                event_payload = {"main": m_filtered}

                self.ws.send(json.dumps(event_payload))
                print(f"[SERVER SUCCESS] Событие отправлено: {m_filtered['m.signature']}")
                return True
            return False
        except Exception as e:
            logger.error(f"Ошибка WebSocket: {e}")
            self.ws = None
            return False