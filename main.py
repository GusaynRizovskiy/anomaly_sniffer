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
from core.cic_ids_processor import CICIDSProcessor

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


def log_anomaly(anomaly_data):
    """
    Запись данных об аномалии в JSON-файл.
    Структура файла будет соответствовать требованиям СИБ.
    """
    try:
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_path = os.path.join(log_dir, f"anomaly_{timestamp}.json")

        with open(log_file_path, 'w') as f:
            json.dump(anomaly_data, f, indent=4)
        logging.info(f"Данные об аномалии записаны в файл: {log_file_path}")
    except Exception as e:
        logging.error(f"Ошибка при записи лога аномалии: {e}")


def handle_metrics_for_collect(metrics):
    """Callback для режима 'collect'."""
    try:
        # Проверка, что файл существует, и запись заголовка, если это новая запись.
        file_exists = os.path.exists(args.data_file)
        headers_with_timestamp = ['timestamp'] + HEADERS

        row_data = [datetime.now().strftime('%Y-%m-%d %H:%M:%S')] + [
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
        df.to_csv(args.data_file, mode='a', header=not file_exists, index=False)
        logging.info(f"Записаны данные за интервал. Всего пакетов: {metrics['total']['packets']}")
    except Exception as e:
        logging.error(f"Ошибка при обработке метрик в режиме 'collect': {e}")


def main():
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(
        description="Программа для обнаружения сетевых аномалий с помощью автокодировщика.")
    parser.add_argument("mode", choices=['train', 'test', 'collect', 'dataset-train', 'dataset-test'],
                        help="Режим работы программы: 'train' (обучение на реальном трафике), 'test' (тестирование на реальном трафике), 'collect' (сбор данных), 'dataset-train' (обучение на датасете), 'dataset-test' (тестирование на датасете).")
    parser.add_argument("--interface", help="Сетевой интерфейс для захвата трафика (например, 'eth0').")
    parser.add_argument("--network", help="CIDR-адрес сети для фильтрации трафика (например, '192.168.1.0/24').")
    parser.add_argument("--interval", type=int, default=10,
                        help="Интервал агрегации метрик в секундах. Используется в режиме 'test' и 'collect'.")
    parser.add_argument("--model-dir", default="models", help="Директория для сохранения/загрузки модели.")
    parser.add_argument("--time-step", type=int, default=10,
                        help="Размер временного шага для последовательностей. Используется в обоих режимах.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Количество эпох для обучения. Используется только в режиме 'train' и 'dataset-train'.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Размер пакета для обучения. Используется только в режиме 'train' и 'dataset-train'.")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Порог для определения аномалий. Используется в режиме 'test' и 'dataset-test'.")
    parser.add_argument("--data-file",
                        help="Путь к файлу с данными. В режиме 'train' для чтения, в 'collect' для записи, в 'dataset-train' для обучения.")
    parser.add_argument("--test-data-file",
                        help="Путь к файлу с данными для тестирования. Используется в режиме 'dataset-test'.")

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "autoencoder_model.h5")
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")

    detector = AnomalyDetector(time_step=args.time_step)
    processor = DataProcessor()

    os.makedirs(args.model_dir, exist_ok=True)

    if args.mode == 'train':
        if not args.data_file or not os.path.exists(args.data_file):
            logging.error("В режиме 'train' необходимо указать путь к файлу с данными для обучения (--data-file).")
            return

        logging.info("Запуск программы в режиме обучения...")
        try:
            X_train = processor.load_and_preprocess_training_data(args.data_file)
            if X_train is None:
                return

            X_train_sequences = processor.create_sequences(X_train, args.time_step)

            detector.build_model()
            detector.train(X_train_sequences, args.epochs, args.batch_size, model_path, scaler_path)
        except Exception as e:
            logging.critical(f"Произошла ошибка в режиме обучения: {e}")

    elif args.mode == 'test':
        logging.info("Запуск программы в режиме тестирования...")

        if not detector.load_model(model_path) or not os.path.exists(scaler_path):
            logging.error("Модель или scaler не найдены. Сначала запустите программу в режиме обучения.")
            return

        with open(scaler_path, 'rb') as f:
            processor.scaler = pickle.load(f)

        data_queue = collections.deque(maxlen=args.time_step)

        def handle_metrics_for_test(metrics):
            nonlocal data_queue

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

            scaled_data = processor.preprocess_data(pd.DataFrame([row_data], columns=HEADERS))
            # ИСПРАВЛЕНИЕ: Добавляем одномерный массив (строку) в очередь как один элемент
            data_queue.append(scaled_data[0])

            if len(data_queue) == args.time_step:
                # ИСПРАВЛЕНИЕ: Изменяем форму массива для правильного формата (1, time_step, 26)
                input_data = np.array(list(data_queue)).reshape(1, args.time_step, 26)

                reconstruction_error = detector.calculate_reconstruction_error(input_data)
                is_anomaly = reconstruction_error > args.threshold

                status = "АНОМАЛИЯ ОБНАРУЖЕНА" if is_anomaly else "Данные обработаны"
                logging.info(f"Статус: {status} | Ошибка реконструкции: {reconstruction_error:.4f}")

                if is_anomaly:
                    anomaly_event = {
                        "gid": 1,
                        "sid": 0,
                        "rev": 0,
                        "signature_msg": f"Anomaly detected on live traffic. Reconstruction error: {reconstruction_error:.4f}",
                        "appearance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "priority": 1,
                        "source_ip": "N/A",
                        "source_port": "N/A",
                        "destination_ip": "N/A",
                        "destination_port": "N/A",
                        "packet_dump": ""
                    }
                    log_anomaly(anomaly_event)

        sniffer = Sniffer(
            interface=args.interface,
            network_cidr=args.network,
            time_interval=args.interval,
            callback=handle_metrics_for_test
        )
        sniffer.start_sniffing()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nПрограмма завершена пользователем.")
            sniffer.stop_sniffing()

    elif args.mode == 'collect':
        if not args.interface or not args.network or not args.data_file:
            logging.error("В режиме 'collect' необходимо указать интерфейс, сеть и файл для сохранения данных.")
            return

        logging.info(f"Запуск программы в режиме сбора данных на интерфейсе {args.interface}...")

        # Удаляем файл, если он уже существует, для чистой записи
        if os.path.exists(args.data_file):
            os.remove(args.data_file)

        sniffer = Sniffer(
            interface=args.interface,
            network_cidr=args.network,
            time_interval=args.interval,
            callback=handle_metrics_for_collect
        )
        sniffer.start_sniffing()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nПрограмма завершена пользователем.")
            sniffer.stop_sniffing()

    elif args.mode == 'dataset-train':
        if not args.data_file or not os.path.exists(args.data_file):
            logging.error("В режиме 'dataset-train' необходимо указать путь к файлу с данными (--data-file).")
            return

        logging.info(f"Запуск программы в режиме обучения на датасете: {args.data_file}")

        try:
            dataset_processor = CICIDSProcessor()
            raw_data = dataset_processor.load_and_preprocess_training_data(args.data_file)

            if raw_data is None:
                return

            X_train = dataset_processor.create_sequences(raw_data, args.time_step)

            detector.build_model()
            detector.train(X_train, args.epochs, args.batch_size, model_path, scaler_path)

            with open(scaler_path, 'wb') as f:
                pickle.dump(dataset_processor.scaler, f)
            logging.info("Scaler успешно сохранен.")

        except Exception as e:
            logging.critical(f"Произошла ошибка в режиме обучения на датасете: {e}")

    elif args.mode == 'dataset-test':
        if not args.test_data_file or not os.path.exists(args.test_data_file):
            logging.error(
                "В режиме 'dataset-test' необходимо указать путь к файлу с данными для тестирования (--test-data-file).")
            return

        logging.info(f"Запуск программы в режиме тестирования на датасете: {args.test_data_file}")

        if not detector.load_model(model_path) or not os.path.exists(scaler_path):
            logging.error("Модель или scaler не найдены. Сначала запустите программу в режиме обучения на датасете.")
            return

        with open(scaler_path, 'rb') as f:
            scaler_from_file = pickle.load(f)

        try:
            dataset_processor = CICIDSProcessor(scaler_from_file)
            raw_data = dataset_processor.load_and_preprocess_training_data(args.test_data_file, fit_scaler=False)
            if raw_data is None:
                return

            X_test = dataset_processor.create_sequences(raw_data, args.time_step)

            for i in range(len(X_test)):
                input_data = X_test[i:i + 1]
                reconstruction_error = detector.calculate_reconstruction_error(input_data)
                is_anomaly = reconstruction_error > args.threshold

                status = "АНОМАЛИЯ ОБНАРУЖЕНА" if is_anomaly else "Данные обработаны"
                logging.info(f"Статус: {status} | Ошибка реконструкции: {reconstruction_error:.4f}")

                if is_anomaly:
                    anomaly_event = {
                        "gid": 1,
                        "sid": 0,
                        "rev": 0,
                        "signature_msg": f"Anomaly detected on test dataset. Reconstruction error: {reconstruction_error:.4f}",
                        "appearance_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "priority": 1,
                        "source_ip": "N/A",
                        "source_port": "N/A",
                        "destination_ip": "N/A",
                        "destination_port": "N/A",
                        "packet_dump": ""
                    }
                    log_anomaly(anomaly_event)

        except Exception as e:
            logging.critical(f"Произошла ошибка в режиме тестирования на датасете: {e}")


if __name__ == "__main__":
    main()