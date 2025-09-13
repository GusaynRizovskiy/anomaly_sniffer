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

        return scaled_data.flatten().reshape(-1, 1)

    def create_sequences(self, data, time_step):
        """Создание последовательностей для нейронной сети."""
        xs = []
        for i in range(len(data) - time_step):
            xs.append(data[i:(i + time_step), 0])
        return np.array(xs)

    def load_and_preprocess_training_data(self, file_path, fit_scaler=True):
        """Загрузка и нормализация данных для обучения."""
        try:
            data = pd.read_csv(file_path, delimiter=',')
            if data.shape[1] <= 1:
                logger.error("Не удалось загрузить данные из файла. Убедитесь, что файл имеет правильный формат CSV.")
                return None

            # Удаляем столбец с датой/временем, если он есть
            data = data.iloc[:, 1:]

            # Нормализация
            if fit_scaler or self.scaler is None:
                self.scaler = MinMaxScaler()
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)

            return scaled_data
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            return None