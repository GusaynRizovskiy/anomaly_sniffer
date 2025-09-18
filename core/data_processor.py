# anomaly_sniffer/core/data_processor.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataProcessor:
    def __init__(self, scaler=None):
        self.scaler = scaler

    def preprocess_data(self, data):
        """Предварительная обработка и нормализация данных."""
        if not isinstance(data, pd.DataFrame):
            # Предполагаем, что данные приходят в виде списка или массива,
            # и создаем DataFrame для удобства.
            # Если данные - это одна строка, то data[0] будет первым элементом,
            # и мы получим правильное количество колонок.
            if len(data.shape) == 1:
                data = data.reshape(1, -1)
            data = pd.DataFrame(data, columns=[f'metric_{i}' for i in range(data.shape[1])])

        # Удаляем временной столбец, если он есть
        if data.columns[0].lower().startswith('time'):
            data = data.iloc[:, 1:]

        # Нормализация
        if self.scaler is None:
            self.scaler = MinMaxScaler()
            scaled_data = self.scaler.fit_transform(data)
        else:
            scaled_data = self.scaler.transform(data)

        # reshape(-1, 1) изменяет форму массива, чтобы он имел один столбец.
        # Это нужно для совместимости с входным слоем LSTM.
        return scaled_data.flatten().reshape(-1, 1)

    def create_sequences(self, data, time_step):
        """Создание последовательностей для нейронной сети."""
        xs = []
        for i in range(len(data) - time_step):
            # Изменен, чтобы работать с нормализованным np.array
            xs.append(data[i:(i + time_step)])
        # Изменяем размерность для соответствия модели LSTM
        return np.array(xs).reshape(-1, time_step, 1)

    def load_and_preprocess_training_data(self, file_path, fit_scaler=True, columns_to_use=None):
        """Загрузка и нормализация данных для обучения."""
        try:
            data = pd.read_csv(file_path, delimiter=',')
            if data.shape[1] <= 1:
                logger.error("Не удалось загрузить данные из файла. Убедитесь, что файл имеет правильный формат CSV.")
                return None

            # Удаление столбца с датой/временем, если он есть
            if 'Timestamp' in data.columns:
                data = data.drop('Timestamp', axis=1)

            # Выбор только нужных столбцов, если они указаны
            if columns_to_use:
                data = data[columns_to_use]

            # Удаление строк с бесконечными значениями или NaN
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)

            # Нормализация
            if fit_scaler or self.scaler is None:
                self.scaler = self.scaler or MinMaxScaler()
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)

            return scaled_data
        except Exception as e:
            logger.error(f"Ошибка при загрузке файла: {e}")
            return None