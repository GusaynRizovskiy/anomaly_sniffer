# -*- coding: utf-8 -*-
import json
import requests
import websocket
import logging
import os
from datetime import datetime
from dotenv import load_dotenv

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

    def authenticate(self):
        """Получение accessToken через REST API."""
        try:
            url = f"{self.base_url}/api/auth/login"
            payload = {
                "login": self.login,
                "password": self.password,
                "type": "sensor-user"
            }
            # verify=False для самоподписанных сертификатов
            response = requests.post(url, json=payload, verify=False, timeout=5)
            if response.status_code == 200:
                self.token = response.json().get('accessToken')
                logger.info("Успешная аутентификация на удаленном сервере.")
                return True
            else:
                logger.error(f"Ошибка аутентификации: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"Не удалось связаться с сервером аутентификации: {e}")
            return False

    def connect_ws(self):
        """Установка WebSocket соединения."""
        if not self.token:
            if not self.authenticate():
                return

        # Формируем URL динамически на основе base_url
        ws_url = f"wss://{self.raw_url}/integrated-container-ids/connection-integrated-container-ids?token={self.token}"
        try:
            self.ws = websocket.create_connection(
                ws_url,
                sslopt={"cert_reqs": websocket.ssl.CERT_NONE},
                timeout=5
            )
            logger.info("WebSocket соединение установлено.")
        except Exception as e:
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
                event = {
                    "type": "integratedContainerIds/transmittingEvents",
                    "transmittingEvents": [
                        {
                            "event_type": "alert",
                            "timestamp": datetime.now().isoformat(),
                            "src_ip": ctx.get('src_ip', '0.0.0.0'),
                            "src_port": int(ctx.get('src_port', 0)),
                            "dest_ip": ctx.get('dst_ip', '0.0.0.0'),
                            "dest_port": int(ctx.get('dst_port', 0)),
                            "proto": ctx.get('protocol', 'TCP'),
                            "signature": f"Anomaly Detected (Score: {internal_anomaly_data.get('anomaly_score', 0)}%)",
                            "severity": severity_map.get(level, 1),
                            "category": "Network Anomaly"
                        }
                    ]
                }
                # ИНФОРМАТИВНОЕ СООБЩЕНИЕ ДЛЯ КОНСОЛИ
                print(
                    f"[SERVER SUCCESS] [{datetime.now().strftime('%H:%M:%S')}] Событие успешно передано на SIEM-сервер.")

                logger.info("Событие успешно отправлено на сервер.")
                self.ws.send(json.dumps(event))
                logger.info("Событие успешно отправлено на сервер.")
                return True # Успешно отправлено
            return False # WebSocket не подключен
        except Exception as e:
            logger.error(f"Ошибка при отправке через WS: {e}")
            self.ws = None
            return False # Произошла ошибка