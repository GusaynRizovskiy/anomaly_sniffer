# validator.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from core.anomaly_detector import AnomalyDetector
from core.data_processor import DataProcessor

# Настройки (укажи свои пути)
ATTACK_FILE = 'attack_data.csv'  # Твой файл с атаками
MODEL_PATH = 'anomaly_detector_model.keras'
SCALER_PATH = 'scaler.pkl'
THRESHOLD_FILE = 'threshold.txt'
TIME_STEP = 10
NUM_FEATURES = 26  # Как в main.py


def validate():
    print("--- Запуск валидации на файле с атаками ---")

    # 1. Загрузка компонент
    processor = DataProcessor()
    detector = AnomalyDetector(time_step=TIME_STEP, num_features=NUM_FEATURES)

    # Загружаем модель
    detector.load_model(MODEL_PATH)
    if detector.model is None: return

    # Загружаем скейлер (Тот самый, от нормального трафика!)
    processor.scaler = detector.load_scaler(SCALER_PATH)
    if processor.scaler is None: return

    # Загружаем порог
    try:
        with open(THRESHOLD_FILE, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Порог (Threshold): {threshold}")
    except:
        print("Ошибка загрузки порога")
        return

    # 2. Загрузка данных атаки
    print(f"Чтение файла {ATTACK_FILE}...")
    try:
        df = pd.read_csv(ATTACK_FILE)
        # Если есть timestamp, удаляем его (обычно 1-й столбец)
        if df.columns[0].lower() == 'timestamp':
            df = df.iloc[:, 1:]
    except Exception as e:
        print(f"Ошибка чтения CSV: {e}")
        return

    # 3. Препроцессинг
    # ВАЖНО: fit_scaler=False, используем уже обученный скейлер
    # Мы имитируем метод load_and_preprocess, но вручную, чтобы не делать fit
    scaled_data = processor.scaler.transform(df)

    # Создаем последовательности (окна)
    X_test = processor.create_sequences(scaled_data, TIME_STEP)
    print(f"Сформировано {X_test.shape[0]} последовательностей для проверки.")

    # 4. Предсказание и расчет ошибки
    # Считаем ошибку для каждого окна
    reconstruction_errors = []

    # Можно прогнать батчем (быстрее)
    reconstructions = detector.model.predict(X_test, verbose=1)

    # Вычисляем MSE для каждого образца вручную (аналог calculate_reconstruction_error)
    # mse = mean((input - output)^2)
    mse = np.mean(np.power(X_test - reconstructions, 2), axis=(1, 2))

    # 5. Анализ результатов
    anomalies_indices = np.where(mse > threshold)[0]
    num_anomalies = len(anomalies_indices)
    total_samples = len(mse)

    print("\n--- Результаты ---")
    print(f"Всего проверено окон: {total_samples}")
    print(f"Обнаружено аномалий: {num_anomalies}")
    print(f"Процент аномалий: {(num_anomalies / total_samples) * 100:.2f}%")

    # 6. Визуализация (Для диплома - супер!)
    plt.figure(figsize=(14, 6))
    plt.plot(mse, label='Ошибка реконструкции (MSE)', color='blue')
    plt.axhline(y=threshold, color='red', linestyle='--', label='Порог (Threshold)')

    # Подсветка зон аномалий
    if num_anomalies > 0:
        plt.scatter(anomalies_indices, mse[anomalies_indices], color='red', s=10, label='Аномалия')

    plt.title('Результат проверки на тестовом датасете')
    plt.xlabel('Временные интервалы (Time Steps)')
    plt.ylabel('MSE (Ошибка)')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    validate()