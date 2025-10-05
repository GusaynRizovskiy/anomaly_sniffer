# -*- coding: utf-8 -*-
# anomaly_sniffer/main.py

import argparse
import os
import numpy as np
import threading
import time
from datetime import datetime
import logging
import pickle
import collections
import pandas as pd
import json

from core.anomaly_detector import AnomalyDetector
from core.sniffer import Sniffer
from core.data_processor import DataProcessor

# Заголовки для DataFrame, чтобы избежать UserWarning при нормализации
HEADERS = [
    'total_packets', 'total_loopback', 'total_multicast', 'total_udp',
    'total_tcp', 'total_options', 'total_fragment', 'total_fin', 'total_syn',
    'total_intensity', 'input_packets', 'input_udp', 'input_tcp',
    'input_options', 'input_fragment', 'input_fin', 'input_syn',
    'input_intensity', 'output_packets', 'output_udp', 'output_tcp',
    'output_options', 'output_fragment', 'output_fin', 'output_syn',
    'output_intensity'
]
NUM_FEATURES = len(HEADERS)  # НОВОЕ: Определяем количество признаков

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Глобальные переменные для режима TEST
data_buffer = collections.deque(maxlen=None)  # Размер будет установлен в main()
threshold = None


def log_anomaly(anomaly_data):
    """
    Запись данных об аномалии в JSON-файл.
    Структура файла будет соответствовать требованиям СИБ.
    """
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)

        # Формируем имя файла на основе текущей даты
        filename = datetime.now().strftime("anomaly_log_%Y-%m-%d.json")
        filepath = os.path.join(log_dir, filename)

        # Подготовка записи
        record = {
            "timestamp": datetime.now().isoformat(),
            "level": "CRITICAL",
            "event_id": "NETWORK_ANOMALY_DETECTED",
            "description": "Обнаружена сетевая аномалия (высокая ошибка реконструкции автокодировщика)",
            "details": anomaly_data
        }

        # Запись в файл с добавлением
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

        logger.warning(f"!!! АНОМАЛИЯ ЗАПИСАНА: {record['description']} | MSE: {anomaly_data['mse_error']:.4f}")

    except Exception as e:
        logger.error(f"Ошибка при записи лога аномалии: {e}")


def handle_metrics_for_test(metrics, processor, detector, args):
    """
    Обработчик метрик для режима 'test' (детекция аномалий).
    """
    global data_buffer, threshold

    if threshold is None:
        logger.warning("Порог не установлен. Сначала необходимо обучить или загрузить модель/порог.")
        return

    # Собираем текущие 26 метрик в DataFrame для нормализации
    row_data = [
        metrics['total']['packets'], metrics['total']['loopback'], metrics['total']['multicast'],
        metrics['total']['udp'], metrics['total']['tcp'], metrics['total']['options'],
        metrics['total']['fragment'], metrics['total']['fin'], metrics['total']['syn'],
        metrics['total']['intensivity'],
        metrics['input']['packets'], metrics['input']['udp'], metrics['input']['tcp'],
        metrics['input']['options'], metrics['input']['fragment'], metrics['input']['fin'],
        metrics['input']['syn'], metrics['input']['intensivity'],
        metrics['output']['packets'], metrics['output']['udp'], metrics['output']['tcp'],
        metrics['output']['options'], metrics['output']['fragment'], metrics['output']['fin'],
        metrics['output']['syn'], metrics['output']['intensivity']
    ]

    df = pd.DataFrame([row_data], columns=HEADERS)

    # Нормализация
    try:
        # data_processor.py изменен, чтобы возвращать 2D массив (1, 26)
        scaled_metric = processor.preprocess_data(df)
    except Exception as e:
        logger.error(f"Ошибка нормализации данных в режиме test: {e}")
        return

    # Добавляем ВЕСЬ ВЕКТОР (26 признаков) в буфер
    # ИЗМЕНЕНО: Добавляем весь массив scaled_metric[0]
    data_buffer.append(scaled_metric[0])

    if len(data_buffer) == args.time_step:
        # Формируем входные данные для модели (1, time_step, num_features)
        # ИЗМЕНЕНО: Используем np.newaxis для добавления размерности батча
        input_data = np.array(list(data_buffer))[np.newaxis, :, :]

        # Вычисляем ошибку реконструкции
        error = detector.calculate_reconstruction_error(input_data)

        # Проверка на аномалию
        if error > threshold:
            # Логирование аномалии
            log_anomaly({
                "mse_error": error,
                "threshold": threshold,
                "metrics_snapshot": dict(zip(HEADERS, row_data))
            })
        else:
            logger.info(f"OK. MSE: {error:.4f} (Threshold: {threshold:.4f})")


def handle_metrics_for_collect(metrics, args):
    """
    Обработчик метрик для режима 'collect' (сбор данных).
    """
    # ... (логика сбора данных оставлена без изменений, т.к. она была корректна)
    headers_with_timestamp = ['timestamp'] + HEADERS

    row_data = [datetime.now().isoformat(),
                metrics['total']['packets'], metrics['total']['loopback'], metrics['total']['multicast'],
                metrics['total']['udp'], metrics['total']['tcp'], metrics['total']['options'],
                metrics['total']['fragment'], metrics['total']['fin'], metrics['total']['syn'],
                metrics['total']['intensivity'],
                metrics['input']['packets'], metrics['input']['udp'], metrics['input']['tcp'],
                metrics['input']['options'], metrics['input']['fragment'], metrics['input']['fin'],
                metrics['input']['syn'], metrics['input']['intensivity'],
                metrics['output']['packets'], metrics['output']['udp'], metrics['output']['tcp'],
                metrics['output']['options'], metrics['output']['fragment'], metrics['output']['fin'],
                metrics['output']['syn'], metrics['output']['intensivity']
                ]

    df = pd.DataFrame([row_data], columns=headers_with_timestamp)
    df.to_csv(args.data_file, mode='a', header=False, index=False)
    logging.info(f"Записаны данные за интервал. Всего пакетов: {metrics['total']['packets']}")


def main():
    global data_buffer, threshold

    # 1. Парсинг аргументов
    parser = argparse.ArgumentParser(description="Сетевой сниффер с детектором аномалий на базе автокодировщика.")
    parser.add_argument('mode', choices=['collect', 'train', 'test'],
                        help="Режим работы: сбор данных, обучение или тестирование.")
    parser.add_argument('-i', '--interface', default='eth0', help="Сетевой интерфейс для прослушивания.")
    parser.add_argument('-n', '--network', default='192.168.1.0/24',
                        help="Локальная сеть в формате CIDR (для определения входящего/исходящего).")
    parser.add_argument('-t', '--interval', type=int, default=5, help="Интервал агрегации метрик в секундах.")
    parser.add_argument('-d', '--data-file', default='training_data.csv', help="Файл для сохранения/загрузки данных.")
    parser.add_argument('-ts', '--time_step', type=int, default=10, help="Длина временной последовательности для LSTM.")
    parser.add_argument('-e', '--epochs', type=int, default=50, help="Количество эпох для обучения.")
    parser.add_argument('-b', '--batch_size', type=int, default=64, help="Размер батча для обучения.")
    parser.add_argument('-m', '--model_path', default='anomaly_detector_model.keras', help="Путь к файлу модели.")
    parser.add_argument('-s', '--scaler_path', default='scaler.pkl', help="Путь к файлу нормализатора.")
    parser.add_argument('-thr', '--threshold_file', default='threshold.txt', help="Путь к файлу с порогом ошибки.")

    args = parser.parse_args()

    # 2. Инициализация процессора
    processor = DataProcessor()

    # 3. Инициализация детектора
    # ИЗМЕНЕНО: Передаем количество признаков в конструктор
    detector = AnomalyDetector(time_step=args.time_step, num_features=NUM_FEATURES)

    if args.mode == 'train':
        # 4. Режим обучения
        logger.info(f"Запуск режима обучения. Загрузка данных из {args.data_file}...")

        # Загрузка и нормализация данных
        raw_data = processor.load_and_preprocess_training_data(args.data_file, fit_scaler=True)
        if raw_data is None:
            return

        # Сохраняем нормализатор
        detector.save_scaler(processor.scaler, args.scaler_path)

        # Создание последовательностей
        # ИЗМЕНЕНО: Передаем 2D массив, а не плоский одномерный ряд. create_sequences теперь работает корректно.
        X_train = processor.create_sequences(raw_data, args.time_step)

        logger.info(f"Форма данных для обучения (samples, time_step, features): {X_train.shape}")

        if X_train.size == 0:
            logger.error(
                "Недостаточно данных для создания последовательностей. Увеличьте интервал сбора данных или уменьшите time_step.")
            return

        # Обучение модели
        detector.train_model(X_train, args.epochs, args.batch_size, args.model_path)

        # Расчет и сохранение порога (на основе ошибки реконструкции обучающего набора)
        # Порог обычно берется как max(MSE) + epsilon или 95-й/99-й перцентиль
        logger.info("Расчет порога ошибки...")
        reconstruction_errors = []
        # Вычисляем ошибку для каждого образца
        for i in range(X_train.shape[0]):
            sample = X_train[i:i + 1]  # Получаем (1, time_step, num_features)
            error = detector.calculate_reconstruction_error(sample)
            reconstruction_errors.append(error)

        reconstruction_errors = np.array(reconstruction_errors)
        new_threshold = np.percentile(reconstruction_errors, 99)  # 99-й перцентиль как порог

        try:
            with open(args.threshold_file, 'w') as f:
                f.write(str(new_threshold))
            threshold = new_threshold
            logger.info(
                f"Порог ошибки (99-й перцентиль) установлен: {threshold:.4f} и сохранен в {args.threshold_file}")
        except Exception as e:
            logger.error(f"Ошибка сохранения порога: {e}")

    elif args.mode == 'test':
        # 5. Режим тестирования/детекции
        logger.info("Запуск режима тестирования. Загрузка модели и нормализатора...")

        # Загрузка модели
        detector.load_model(args.model_path)
        if detector.model is None:
            return

        # Загрузка нормализатора
        processor.scaler = detector.load_scaler(args.scaler_path)
        if processor.scaler is None:
            return

        # Загрузка порога
        try:
            with open(args.threshold_file, 'r') as f:
                threshold = float(f.read().strip())
            logger.info(f"Порог ошибки загружен: {threshold:.4f}")
        except FileNotFoundError:
            logger.error(f"Файл порога не найден: {args.threshold_file}. Сначала запустите режим 'train'.")
            return
        except Exception as e:
            logger.error(f"Ошибка загрузки порога: {e}")
            return

        # Инициализация буфера для хранения time_step векторов по 26 признаков
        data_buffer = collections.deque(maxlen=args.time_step)

        # Запуск сниффера с callback для детекции
        sniffer = Sniffer(
            interface=args.interface,
            network_cidr=args.network,
            time_interval=args.interval,
            callback=lambda metrics: handle_metrics_for_test(metrics, processor, detector, args)
        )
        sniffer.start_sniffing()

        logger.info("Детектор запущен. Ожидание сбора первой последовательности данных...")

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nПрограмма завершена пользователем.")
            sniffer.stop_sniffing()


    elif args.mode == 'collect':
        # 6. Режим сбора данных
        logger.info(f"Запуск режима сбора данных. Запись в {args.data_file} каждые {args.interval} сек.")

        # Добавляем заголовок, если файл не существует
        if not os.path.exists(args.data_file) or os.stat(args.data_file).st_size == 0:
            headers_with_timestamp = ['timestamp'] + HEADERS
            df_header = pd.DataFrame(columns=headers_with_timestamp)
            df_header.to_csv(args.data_file, mode='w', header=True, index=False)
            logger.info(f"Создан новый файл данных {args.data_file} с заголовками.")

        # Запуск сниффера с callback для сбора
        sniffer = Sniffer(
            interface=args.interface,
            network_cidr=args.network,
            time_interval=args.interval,
            callback=lambda metrics: handle_metrics_for_collect(metrics, args)
        )
        sniffer.start_sniffing()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nПрограмма завершена пользователем.")
            sniffer.stop_sniffing()


if __name__ == "__main__":
    main()