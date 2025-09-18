# anomaly_sniffer/core/anomaly_detector.py
# -*- coding: utf-8 -*-
import os
import numpy as np
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Conv1D, LSTM, RepeatVector
from tensorflow.keras.losses import MeanSquaredError as mse_loss
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import logging

logger = logging.getLogger(__name__)


class AnomalyDetector:
    def __init__(self, time_step):
        self.model = None
        self.time_step = time_step
        self.loss_metric = mse_loss()

    def build_model(self,num_features):
        """Создание архитектуры нейронной сети."""
        inputs = Input(shape=(self.time_step, num_features))
        x = Conv1D(filters=32, kernel_size=3, padding='same', activation='relu')(inputs)
        x = LSTM(16, return_sequences=True, activation='relu')(x)
        x = LSTM(8, activation='relu')(x)
        x = RepeatVector(self.time_step)(x)
        x = LSTM(8, return_sequences=True, activation='relu')(x)
        x = LSTM(16, return_sequences=True, activation='relu')(x)
        x = Conv1D(filters=1, kernel_size=3, padding='same', activation='linear')(x)
        self.model = Model(inputs=inputs, outputs=x)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        logger.info("Архитектура модели создана.")

    def train(self, X_train, epochs, batch_size, model_path, scaler_path):
        """Обучение модели."""
        if self.model is None:
            self.build_model()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True)
        ]

        try:
            self.model.fit(
                X_train, X_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.1,
                callbacks=callbacks,
                verbose=2
            )
            logger.info("Обучение завершено. Модель сохранена.")
        except Exception as e:
            logger.error(f"Ошибка в процессе обучения: {e}")
            return

    def calculate_reconstruction_error(self, X):
        """Вычисляет ошибку реконструкции для одного образца."""
        if self.model is None:
            logger.error("Модель не загружена.")
            return 0

        reconstruction = self.model.predict(X, verbose=0)
        mse = self.loss_metric(X, reconstruction).numpy()
        return mse

    def load_model(self, model_path):
        """Загрузка обученной модели."""
        try:
            self.model = load_model(model_path)
            logger.info("Модель успешно загружена.")
            return True
        except Exception as e:
            logger.error(f"Не удалось загрузить модель: {e}")
            return False