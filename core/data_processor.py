# anomaly_sniffer/core/data_processor.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self):
        self.scaler = None

    def preprocess_data(self, data):
        """Предварительная обработка и нормализация данных."""
        if not isinstance(data, pd.DataFrame):
            # Предполагаем, что данные приходят в виде списка или массива,
            # и создаем DataFrame для удобства.
            data = pd.DataFrame(data, columns=[f'metric_{i}' for i in range(len(data[0]))])

        # Удаляем временной столбец, если он есть
        if data.columns[0].lower().startswith('time'):
            data = data.iloc[:, 1:]

        # Нормализация
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)

        # ИЗМЕНЕНО: Возвращаем 2D массив (samples, num_features), а не плоский одномерный ряд
        return scaled_data

    def create_sequences(self, data, time_step):
        """Создание последовательностей для нейронной сети."""
        xs = []
        # data: 2D массив (samples, num_features)
        for i in range(len(data) - time_step + 1):  # +1, чтобы не потерять последний возможный сэмпл
            # ИЗМЕНЕНО: Выбираем все столбцы (:) для последовательности
            xs.append(data[i:(i + time_step), :])

        # Возвращаем 3D массив (samples, time_step, num_features)
        return np.array(xs)

    def load_and_preprocess_training_data(self, file_path, fit_scaler=True):
        """Загрузка и нормализация данных для обучения."""
        try:
            # ИЗМЕНЕНИЕ 1: sep=None и engine='python' позволяют Pandas самому найти разделитель (, или ;)
            data = pd.read_csv(file_path, sep=None, engine='python')

            logger.info(f"Загружен файл: {file_path}. Размер: {data.shape}")

            # Диагностика: если столбцов мало, покажем, что считалось
            if data.shape[1] <= 1:
                logger.error(f"ОШИБКА: Найдено столбцов: {data.shape[1]}. Ожидалось > 1.")
                logger.error(f"Пример данных: {data.head()}")
                logger.error("Проверьте разделитель в CSV файле.")
                return None

            # Удаляем столбец с датой/временем, если он есть (по названию или индексу)
            # Проверяем, похож ли первый столбец на timestamp
            first_col = data.columns[0].lower()
            if 'time' in first_col or 'date' in first_col:
                data = data.iloc[:, 1:]

            # Дополнительная проверка: остались ли данные после удаления времени
            if data.shape[1] == 0:
                logger.error("После удаления метки времени не осталось данных.")
                return None

            # Нормализация
            if fit_scaler or self.scaler is None:
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)

            # Возвращаем 2D массив
            return scaled_data

        except FileNotFoundError:
            logger.error(f"Файл данных не найден: {file_path}")
            return None
        except Exception as e:
            logger.error(f"Ошибка загрузки/обработки данных: {e}")
            return None