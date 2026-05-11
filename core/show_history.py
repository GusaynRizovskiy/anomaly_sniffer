import time
import json
import os
import sys
from datetime import datetime
from prettytable import PrettyTable

# Добавляем текущую директорию в путь, чтобы Python видел папку core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.remote_transmitter import RemoteTransmitter
except ImportError as e:
    print(f"[ERROR] Не удалось импортировать core. Убедитесь, что скрипт в корне проекта. {e}")
    sys.exit(1)


def load_config(config_path="config.json"):
    """Загрузка конфигурации."""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Файл конфигурации {config_path} не найден.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"[ERROR] Ошибка парсинга JSON в {config_path}.")
        sys.exit(1)


def print_attacks_table(attacks):
    """Красивый вывод списка атак в таблицу."""
    os.system('cls' if os.name == 'nt' else 'clear')  # Очистка экрана

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"=== МОНИТОРИНГ АТАК SIEM GuardianX (Обновлено: {now_str}) ===")

    if not attacks:
        print("\n[INFO] Атак за последние 30 минут не обнаружено (или ошибка API).")
        return

    # Создаем таблицу
    table = PrettyTable()
    # Названия колонок (берем из вашей структуры JSON)
    table.field_names = ["ID", "Время (TS)", "Сигнатура", "Severity", "Proto", "Src IP:Port", "Dest IP:Port"]

    # Настройка выравнивания
    table.align["Сигнатура"] = "l"  # По левому краю
    table.align["Src IP:Port"] = "l"
    table.align["Dest IP:Port"] = "l"

    for attack in attacks:
        # Извлекаем данные, безопасно обрабатывая отсутствие ключей
        m = attack.get('main', attack)  # Если структура плоская, берем сам объект

        # Форматируем время для читаемости (если оно в ISO)
        ts = m.get('m.ts', '')
        if ts and 'T' in ts:
            ts = ts.split('T')[1].split('.')[0]  # Оставляем только ЧЧ:ММ:СС

        # Форматируем IP:Port
        src = f"{m.get('m.src_ip', '0.0.0.0')}:{m.get('m.src_port', 0)}"
        dest = f"{m.get('m.dest_ip', '0.0.0.0')}:{m.get('m.dest_port', 0)}"

        severity = m.get('m.severity', 1)
        sev_str = "INFO"
        if severity == 3:
            sev_str = "CRITICAL"
        elif severity == 2:
            sev_str = "WARNING"

        table.add_row([
            m.get('id', 'N/A'),  # ID обычно присваивает БД сервера
            ts,
            m.get('m.signature', 'Unknown')[:30],  # Обрезаем длинные сигнатуры
            sev_str,
            m.get('m.proto', 'TCP'),
            src,
            dest
        ])

    print(table)
    print("\n[CTRL+C для выхода]")


def main():
    config = load_config()

    # Инициализируем передатчик
    # Передаем только URL, логин и пароль. Интерфейс для API не нужен.
    transmitter = RemoteTransmitter(
        base_url=config['network']['server_url'],
        login=config['network']['login'],
        password=config['network']['password'],
        interface=""  # API не нужен интерфейс
    )

    print("[*] Подключение к SIEM API и авторизация...")
    if not transmitter.authenticate():
        print("[FAIL] Не удалось авторизоваться на сервере API.")
        return

    print("[SUCCESS] Авторизация пройдена. Начинаем мониторинг.")
    time.sleep(1)

    # Бесконечный цикл мониторинга
    try:
        while True:
            # Запрашиваем историю за последние 30 минут
            attacks = transmitter.get_attack_history(minutes_back=30)

            # Выводим таблицу
            print_attacks_table(attacks)

            # Ждем 10 секунд перед следующим обновлением
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[*] Мониторинг остановлен пользователем.")


if __name__ == "__main__":
    main()