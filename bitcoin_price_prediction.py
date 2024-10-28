import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# Veri setini yükleme
data = pd.read_csv("/content/consolidated_coin_data.csv", quotechar='"')

# Yalnızca Bitcoin verilerini filtreleme
bitcoin_data = data[data['Currency'].str.lower() == 'bitcoin'].copy()

# Bitcoin verilerinin olup olmadığını kontrol etme
if bitcoin_data.empty:
    print("Veri setinde 'Bitcoin' verisi bulunamadı. Lütfen veri setini kontrol edin.")
else:
    # 'Close' sütununun veri tipini kontrol ederek işlem yap
    if bitcoin_data['Close'].dtype == 'object':
        bitcoin_data['Close'] = bitcoin_data['Close'].str.replace(',', '').astype(float)

    # Tarihe göre sıralama
    bitcoin_data = bitcoin_data.sort_values('Date')

    # Sadece kapanış fiyatlarını alalım
    dataset = bitcoin_data[['Close']].values

    # Veriyi ölçeklendirme
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)

    # Eğitim ve test seti ayrımı
    train_size = int(len(scaled_data) * 0.8)
    test_size = len(scaled_data) - train_size
    train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

    # Zaman serisi verisi hazırlama fonksiyonu
    def create_dataset(dataset, time_step=1):
        X, Y = [], []
        for i in range(len(dataset) - time_step - 1):
            a = dataset[i:(i + time_step), 0]
            X.append(a)
            Y.append(dataset[i + time_step, 0])
        return np.array(X), np.array(Y)

    # Zaman adımını belirleme
    time_step = 120  # Zaman adımını artırarak daha fazla geçmiş veri kullanıyoruz
    X_train, y_train = create_dataset(train_data, time_step)
    X_test, y_test = create_dataset(test_data, time_step)

    # LSTM için giriş şekline uygun boyuta getirme
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # LSTM modeli oluşturma
    model = Sequential()
    model.add(LSTM(units=300, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.3))
    model.add(LSTM(units=200, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(units=100, return_sequences=False))
    model.add(Dropout(0.3))
    model.add(Dense(units=50))
    model.add(Dense(units=1))

    # Modelin derlenmesi (daha düşük öğrenme oranı)
    optimizer = Adam(learning_rate=0.0001)  # Öğrenme oranını daha da düşürdük
    model.compile(optimizer=optimizer, loss='mean_squared_error')

    # Erken durdurma
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    # Modelin eğitilmesi
    history = model.fit(X_train, y_train, batch_size=32, epochs=150, validation_split=0.1, callbacks=[early_stop])

    # Tahminleri ters ölçeklendirme
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Gerçek y_test değerlerini ters ölçeklendirme
    y_test = scaler.inverse_transform([y_test])[0]

    # Performans metrikleri
    rmse = np.sqrt(mean_squared_error(y_test, test_predict[:, 0]))
    mae = mean_absolute_error(y_test, test_predict[:, 0])
    print(f"Root Mean Squared Error (RMSE): {rmse}")
    print(f"Mean Absolute Error (MAE): {mae}")

    # Gerçek ve tahmin edilen fiyatları görselleştirme
    plt.figure(figsize=(14,6))
    plt.plot(scaler.inverse_transform(scaled_data), label='Gerçek Fiyat')
    plt.plot(np.arange(time_step, len(train_predict)+time_step), train_predict, label='Eğitim Tahminleri')
    plt.plot(np.arange(len(train_predict) + (2 * time_step), len(train_predict) + (2 * time_step) + len(test_predict)), test_predict, label='Test Tahminleri')
    plt.legend()
    plt.show()