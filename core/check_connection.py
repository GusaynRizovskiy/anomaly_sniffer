import json
import logging
from remote_transmitter import RemoteTransmitter

def test_config():
    try:
        with open('config.json', 'r') as f:
            cfg = json.load(f)['server']

        base_url = f"https://{cfg['ip']}:{cfg['port']}"
        client = RemoteTransmitter(base_url, cfg['login'], cfg['password'], cfg['type'])

        print(f"--- Проверка подключения к {base_url} ---")
        if client.authenticate():
            print("[SUCCESS] Аутентификация пройдена успешно.")
            client.connect_ws()
            if client.ws and client.ws.connected:
                print("[SUCCESS] WebSocket соединение установлено.")
                client.ws.close()
            else:
                print("[FAIL] Не удалось поднять WebSocket.")
        else:
            print("[FAIL] Ошибка на этапе логина. См. логи выше.")

    except FileNotFoundError:
        print("[ERROR] Файл config.json не найден.")
    except Exception as e:
        print(f"[ERROR] Ошибка при чтении конфига: {e}")


if __name__ == "__main__":
    test_config()