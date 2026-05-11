import time
import json
import os
import sys
from datetime import datetime
from prettytable import PrettyTable

# Добавляем текущую директорию в путь поиска, чтобы Python видел папку core
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from core.remote_transmitter import RemoteTransmitter
except ImportError as e:
    print(f"[ERROR] Не удалось импортировать core. Убедитесь, что скрипт находится в корне проекта. {e}")
    sys.exit(1)


def load_config(config_path="config.json"):
    """Загрузка конфигурационного файла."""
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
    # Оставляем очистку экрана закомментированной для сохранения логов отладки.
    # Если вы хотите, чтобы экран очищался при каждом обновлении, раскомментируйте строку ниже:
    # os.system('cls' if os.name == 'nt' else 'clear')

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n" + "=" * 95)
    print(f"=== МОНИТОРИНГ АТАК SIEM GuardianX (Обновлено: {now_str}) ===")
    print("=" * 95)

    if not attacks:
        print("\n[INFO] Атак за последние 30 минут не обнаружено (или пустой ответ API).")
        return

    # Создаем таблицу
    table = PrettyTable()
    table.field_names = ["ID", "Время (TS)", "Сигнатура", "Приоритет", "Протокол", "Источник (IP:Порт)",
                         "Назначение (IP:Порт)"]

    # Настройка выравнивания колонок
    table.align["Сигнатура"] = "l"  # По левому краю
    table.align["Источник (IP:Порт)"] = "l"
    table.align["Назначение (IP:Порт)"] = "l"

    for attack in attacks:
        # Так как структура ответа плоская, данные забираем напрямую из объекта 'attack'

        # 1. Форматирование времени (извлекаем ЧЧ:ММ:СС из ISO-строки "2026-05-11T13:22:02.056Z")
        ts_raw = attack.get('appearance_time', '')
        ts = 'N/A'
        if ts_raw and 'T' in ts_raw:
            try:
                ts = ts_raw.split('T')[1].split('.')[0]
            except Exception:
                ts = ts_raw

        # 2. Форматирование IP-адресов и портов источника и назначения
        src_ip = attack.get('source_ip', '0.0.0.0')
        src_port = attack.get('source_port', '0')
        src = f"{src_ip}:{src_port}"

        dest_ip = attack.get('destination_ip', '0.0.0.0')
        dest_port = attack.get('destination_port', '0')
        dest = f"{dest_ip}:{dest_port}"

        # 3. Маппинг уровней приоритета (поле 'priority')
        priority_raw = str(attack.get('priority', '1'))
        if priority_raw == '1':
            priority_str = "CRITICAL"
        elif priority_raw == '2':
            priority_str = "WARNING"
        elif priority_raw == '3':
            priority_str = "INFO"
        else:
            priority_str = f"LOW ({priority_raw})"

        # 4. Добавление собранной строки в таблицу
        table.add_row([
            attack.get('id', 'N/A'),
            ts,
            attack.get('signature_msg', 'Unknown')[:45],  # Ограничиваем длину сигнатуры до 45 символов для читаемости
            priority_str,
            attack.get('proto', 'TCP'),
            src,
            dest
        ])

    print(table)
    print("\n[CTRL+C для выхода]")


def main():
    # 1. Загружаем конфигурацию
    config = load_config()
    srv_cfg = config.get('server', {})

    # 2. Формируем базовый URL
    ip = srv_cfg.get('ip')
    port = srv_cfg.get('port')
    if not ip or not port:
        print("[ERROR] В config.json отсутствуют 'ip' или 'port' в секции 'server'.")
        return

    base_url = f"https://{ip}:{port}"

    # 3. Инициализируем передатчик
    transmitter = RemoteTransmitter(
        base_url=base_url,
        login=srv_cfg.get('login'),
        password=srv_cfg.get('password'),
        user_type=srv_cfg.get('type')
    )

    print(f"[*] Подключение к SIEM API ({base_url}) и авторизация...")

    # 4. Проходим аутентификацию
    if not transmitter.authenticate():
        print("[FAIL] Не удалось авторизоваться на сервере API. Проверьте параметры в config.json.")
        return

    print("[SUCCESS] Авторизация пройдена. Начинаем мониторинг истории атак.")
    time.sleep(1)

    # 5. Бесконечный цикл мониторинга
    try:
        while True:
            # Запрашиваем историю атак за последние 30 минут
            attacks = transmitter.get_attack_history(minutes_back=30)

            # Выводим обновленные данные
            print_attacks_table(attacks)

            # Опрашиваем API раз в 10 секунд
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[*] Мониторинг остановлен пользователем.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Произошла ошибка в цикле мониторинга: {e}")


if __name__ == "__main__":
    main()