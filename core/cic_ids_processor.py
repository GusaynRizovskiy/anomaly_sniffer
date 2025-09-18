# anomaly_sniffer/core/cic_ids_processor.py
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from .data_processor import DataProcessor
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class CICIDSProcessor(DataProcessor):
    def __init__(self, scaler=None):
        super().__init__(scaler=scaler)

        # Определяем список столбцов из CIC-IDS, которые будем использовать.
        # Это должен быть поднабор, который можно сопоставить с метриками
        # сетевого трафика.
        self.columns_to_use = [
            ' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets',
            ' Total Length of Fwd Packets', ' Total Length of Bwd Packets',
            ' Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags',
            ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length',
            ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min',
            ' Fwd IAT Total', ' Bwd IAT Total', ' Fwd Packets/s',
            ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length',
            ' Packet Length Mean', ' Packet Length Std',
            ' RST Flag Count', ' SYN Flag Count', ' FIN Flag Count'
        ]

    def load_and_preprocess_training_data(self, file_path, fit_scaler=True):
        """
        Загрузка и предварительная обработка данных CIC-IDS2017.
        Переопределяем метод, чтобы использовать свой набор столбцов.
        """
        logger.info(f"Загрузка и обработка файла {file_path}...")
        try:
            data = pd.read_csv(file_path)

            # Удаляем столбцы, которые не будем использовать
            data = data[self.columns_to_use]

            # Заменяем бесконечные значения на NaN и удаляем строки с NaN
            data.replace([np.inf, -np.inf], np.nan, inplace=True)
            data.dropna(inplace=True)

            # Нормализация
            if fit_scaler or self.scaler is None:
                self.scaler = self.scaler or MinMaxScaler()
                scaled_data = self.scaler.fit_transform(data)
            else:
                scaled_data = self.scaler.transform(data)

            # Изменяем размерность для соответствия модели
            scaled_data = scaled_data.reshape(-1, len(self.columns_to_use))

            return scaled_data

        except Exception as e:
            logger.error(f"Ошибка при загрузке CIC-IDS2017 файла: {e}")
            return None