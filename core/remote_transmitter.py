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
        """
        Запрашивает с сервера ретроспективу атак через HTTP API.
        Возвращает список атак или пустой список при ошибке.
        """
        # 1. Проверка токена
        if not self.token:
            # Пытаемся авторизоваться, если токена нет
            if not self.authenticate():
                logger.error("API History: Нет токена и не удалось авторизоваться.")
                return []

        # 2. Настройка URL и заголовков
        api_url = f"{self.base_url}/api/charts/get-attack-retrospective"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        # 3. Формирование времени (UTC)
        # Сервер ждет формат ISO 8601 с 'Z' в конце
        now = datetime.utcnow()
        start_time = now - timedelta(minutes=minutes_back)

        # 4. Формирование тела запроса (BODY) по вашей спецификации
        payload = {
            "filters": {
                "unitType": "minute",
                "unitValue": str(minutes_back),
                "tsEnd": now.isoformat() + "Z",  # Время КОНЦА
                "tsStart": start_time.isoformat() + "Z",  # Время НАЧАЛА
                "groupSensorsIDSelected": []
            },
            "params": {
                "sortBy": "id",
                "sortDir": "DESC",  # DESC - чтобы новые были сверху
                "page": 1,
                "pageSize": 20,  # Берем последние 20
                "search": ""
            }
        }

        try:
            # 5. Делаем POST запрос
            response = requests.post(
                api_url,
                json=payload,
                headers=headers,
                verify=False,  # Игнорируем самоподписанный сертификат
                timeout=10
            )

            if response.status_code == 200:
                data = response.json()
                # Обычно данные лежат в ключе, например 'items', 'data' или 'content'.
                # Исходя из params.page, сервер должен возвращать структуру с пагинацией.
                # Предположим, список лежит в data['items'] или в корне ответа.

                # Попробуем найти список атак
                attacks = []
                if isinstance(data, list):
                    attacks = data
                elif isinstance(data, dict):
                    attacks = data.get('items', data.get('data', []))

                return attacks

            elif response.status_code == 401:
                logger.warning("API History: Токен протух, сбрасываем.")
                self.token = None  # Сбрасываем токен, чтобы при следующем вызове переавторизоваться
                return []
            else:
                logger.error(f"API History Ошибка: {response.status_code}, Ответ: {response.text}")
                return []

        except Exception as e:
            logger.error(f"Исключение при запросе к API истории: {e}")
            return []

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
        """Преобразование внутреннего формата в формат сервера и отправка. Возвращает True/False."""
        # 1. Если нет токена (не прошли аутентификацию), сразу выходим
        if not self.token:
            return False

        try:
            if not self.ws or not hasattr(self.ws, 'connected') or not self.ws.connected:
                self.connect_ws()

            if self.ws and self.ws.connected:
                severity_map = {"CRITICAL": 3, "WARNING": 2, "INFO": 1}
                level = self._get_severity_local(
                    internal_anomaly_data['mse_error'],
                    internal_anomaly_data['threshold']
                )

                ctx = internal_anomaly_data.get('network_context', {})

                # Формируем структуру как на сервере
                # ==========================================================
                # ФОРМИРУЕМ НОВУЮ СТРУКТУРУ ДАННЫХ (ОБНОВЛЕНО)
                # Поля, которые мы не знаем (gid, rev, flow_id и т.д.),
                # согласно рекомендации, оставляем пустыми (None или "").
                # ==========================================================
                event_payload = {
                    "main": {
                        "m.gid": None,  # Неизвестно сенсору
                        "m.signature_id": 1,  # ID типа сигнатуры (можно зашить 1 для аномалий)
                        "m.rev": None,  # Ревизия (неизвестно)
                        "m.category": "Network Anomaly",
                        "m.signature": f"Anomaly Detected (Score: {internal_anomaly_data.get('anomaly_score', 0)}%)",
                        "m.ts": datetime.now().isoformat(),  # Текущее время
                        "m.flow_id": None,  # ID потока (неизвестно без DPI)
                        "m.severity": severity_map.get(level, 1),
                        "m.proto": ctx.get('protocol', 'TCP').upper(),
                        "m.src_ip": ctx.get('src_ip', '0.0.0.0'),
                        "m.src_port": int(ctx.get('src_port', 0)),
                        "m.dest_ip": ctx.get('dst_ip', '0.0.0.0'),  # Обратите внимание: dest_ip, а не dst_ip
                        "m.dest_port": int(ctx.get('dst_port', 0))
                    }
                }
                # Отправляем JSON
                self.ws.send(json.dumps(event_payload))
                # ==========================================================
                # ИНФОРМАТИВНОЕ СООБЩЕНИЕ ДЛЯ КОНСОЛИ
                print(
                    f"[SERVER SUCCESS] [{datetime.now().strftime('%H:%M:%S')}] Событие успешно передано на SIEM-сервер.")
                logger.info("Событие успешно отправлено на сервер.")
                return True # Успешно отправлено
            return False # WebSocket не подключен
        except Exception as e:
            logger.error(f"Ошибка при отправке через WS: {e}")
            self.ws = None
            return False # Произошла ошибка