import json
import time
import os
import logging
from remote_transmitter import RemoteTransmitter

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_config(config_path='config.json'):
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def run_sender():
    # Загрузка конфига
    with open('config.json', 'r') as f:
        config = json.load(f)

    srv = config['server']
    setts = config['settings']

    transmitter = RemoteTransmitter(
        base_url=f"https://{srv['ip']}:{srv['port']}",
        login=srv['login'],
        password=srv['password'],
        user_type=srv['type']
    )

    log_dir = setts['log_file_path']
    interval = setts['scan_interval_seconds']

    print(f"[*] Мониторинг директории: {log_dir}")

    while True:
        if os.path.exists(log_dir):
            # Список всех json файлов в папке
            files = [f for f in os.listdir(log_dir) if f.endswith('.json')]

            if files and transmitter.authenticate():
                for file_name in files:
                    file_path = os.path.join(log_dir, file_name)
                    try:
                        with open(file_path, 'r') as f:
                            event_data = json.load(f)

                        # Отправка данных через WebSocket или метод класса
                        if transmitter.send_event(event_data):
                            print(f"[OK] Файл {file_name} отправлен.")

                            # Если в конфиге delete_after_send: true
                            if setts.get('delete_after_send'):
                                os.remove(file_path)
                            else:
                                # Чтобы не отправлять одно и то же, можно переносить в архив
                                archive_dir = os.path.join(log_dir, "sent")
                                os.makedirs(archive_dir, exist_ok=True)
                                os.rename(file_path, os.path.join(archive_dir, file_name))

                    except Exception as e:
                        print(f"[ERROR] Не удалось обработать {file_name}: {e}")

        time.sleep(interval)


if __name__ == "__main__":
    run_sender()