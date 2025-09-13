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

from core.anomaly_detector import AnomalyDetector
from core.sniffer import Sniffer
from core.data_processor import DataProcessor


def main():
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    parser = argparse.ArgumentParser(
        description="Программа для обнаружения сетевых аномалий с помощью автокодировщика.")
    parser.add_argument("mode", choices=['train', 'test'],
                        help="Режим работы программы: 'train' для обучения, 'test' для онлайн-тестирования.")
    parser.add_argument("--interface", help="Сетевой интерфейс для захвата трафика (например, 'eth0').")
    parser.add_argument("--network", help="CIDR-адрес сети для фильтрации трафика (например, '192.168.1.0/24').")
    parser.add_argument("--interval", type=int, default=10,
                        help="Интервал агрегации метрик в секундах. Используется в режиме 'test'.")
    parser.add_argument("--model-dir", default="models", help="Директория для сохранения/загрузки модели.")
    parser.add_argument("--time-step", type=int, default=10,
                        help="Размер временного шага для последовательностей. Используется в обоих режимах.")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Количество эпох для обучения. Используется только в режиме 'train'.")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Размер пакета для обучения. Используется только в режиме 'train'.")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Порог для определения аномалий. Используется в режиме 'test'.")
    parser.add_argument("--data-file",
                        help="Путь к файлу с данными для обучения. Используется только в режиме 'train'.")

    args = parser.parse_args()

    model_path = os.path.join(args.model_dir, "autoencoder_model.h5")
    scaler_path = os.path.join(args.model_dir, "scaler.pkl")

    detector = AnomalyDetector(time_step=args.time_step)
    processor = DataProcessor()

    os.makedirs(args.model_dir, exist_ok=True)

    if args.mode == 'train':
        if not args.data_file or not os.path.exists(args.data_file):
            logging.error("В режиме 'train' необходимо указать путь к файлу с данными (--data-file).")
            return

        logging.info("Запуск программы в режиме обучения...")

        try:
            raw_data = processor.load_and_preprocess_training_data(args.data_file)
            if raw_data is None:
                return

            X_train = processor.create_sequences(raw_data.flatten().reshape(-1, 1), args.time_step)

            detector.build_model()
            detector.train(X_train, args.epochs, args.batch_size, model_path, scaler_path)

            with open(scaler_path, 'wb') as f:
                pickle.dump(processor.scaler, f)
            logging.info("Scaler успешно сохранен.")

        except Exception as e:
            logging.critical(f"Произошла ошибка в режиме обучения: {e}")

    elif args.mode == 'test':
        logging.info("Запуск программы в режиме тестирования аномалий...")
        if not detector.load_model(model_path) or not os.path.exists(scaler_path):
            logging.error("Модель или scaler не найдены. Сначала запустите программу в режиме обучения.")
            return

        with open(scaler_path, 'rb') as f:
            processor.scaler = pickle.load(f)

        data_buffer = collections.deque(maxlen=args.time_step)
        buffer_lock = threading.Lock()

        def handle_metrics(metrics):
            """Обработчик агрегированных метрик от сниффера."""
            nonlocal processor
            nonlocal detector

            # Здесь мы преобразуем словарь метрик в список для обработки
            # Порядок метрик должен соответствовать тому, что было на обучении
            # В данном примере, я беру 'packets', 'udp', 'tcp', 'options', 'fragment', 'fin', 'sin' из секции 'total'
            # (необходимо убедиться, что список метрик соответствует тому, что было в файле для обучения)
            metric_values = [
                metrics['total']['packets'], metrics['total']['loopback'], metrics['total']['multicast'],
                metrics['total']['udp'], metrics['total']['tcp'], metrics['total']['options'],
                metrics['total']['fragment'], metrics['total']['fin'], metrics['total']['sin'],
                metrics['total']['intensivity'],
                metrics['input']['packets'], metrics['input']['udp'], metrics['input']['tcp'],
                metrics['input']['options'], metrics['input']['fragment'], metrics['input']['fin'],
                metrics['input']['sin'], metrics['input']['intensivity'],
                metrics['output']['packets'], metrics['output']['udp'], metrics['output']['tcp'],
                metrics['output']['options'], metrics['output']['fragment'], metrics['output']['fin'],
                metrics['output']['sin'], metrics['output']['intensivity']
            ]

            with buffer_lock:
                # Нормализация одной группы метрик
                scaled_metric = processor.scaler.transform(np.array(metric_values).reshape(1, -1))
                data_buffer.append(scaled_metric.flatten()[0])

                if len(data_buffer) == args.time_step:
                    input_data = np.array(list(data_buffer)).reshape(1, args.time_step, 1)
                    reconstruction_error = detector.calculate_reconstruction_error(input_data)
                    is_anomaly = reconstruction_error > args.threshold

                    status = "АНОМАЛИЯ ОБНАРУЖЕНА" if is_anomaly else "Данные обработаны"
                    logging.info(f"Статус: {status} | Ошибка реконструкции: {reconstruction_error:.4f}")
                    data_buffer.popleft()  # Удаляем старый элемент, чтобы буфер оставался скользящим

        # Запуск сниффера в отдельном потоке
        sniffer = Sniffer(
            interface=args.interface,
            network_cidr=args.network,
            time_interval=args.interval,
            callback=handle_metrics
        )
        sniffer.start_sniffing()

        # Главный поток ждет завершения
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("\nПрограмма завершена пользователем.")
            sniffer.stop_sniffing()


if __name__ == "__main__":
    main()