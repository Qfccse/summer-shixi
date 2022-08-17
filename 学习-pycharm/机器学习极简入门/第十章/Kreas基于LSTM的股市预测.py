import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler
from keras import regularizers

google_stock = pd.read_excel('道琼斯综合.xls')
print(google_stock.tail())
print(google_stock.head())

time_stamp = 50
google_stock = google_stock[['开盘价(元/点)_OpPr']]
train = google_stock[0:7000 + time_stamp]
valid = google_stock[7000 - time_stamp:8500 + time_stamp]
test = google_stock[8500 - time_stamp:]


print(train.head())

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(train)
x_train, y_train = [], []

for i in range(time_stamp, len(train) - 5):
    x_train.append(scaled_data[i - time_stamp:i])
    y_train.append(scaled_data[i:i + 5])

#print(x_train[0:10])
#print(y_train[0:10])
x_train, y_train = np.array(x_train), np.array(y_train).reshape(-1, 5)

#print(y_train[0:10])

scaled_data = scaler.fit_transform(valid)
x_valid, y_valid = [], []
for i in range(time_stamp, len(valid) - 5):
    x_valid.append(scaled_data[i - time_stamp:i])
    y_valid.append(scaled_data[i: i + 5])
x_valid, y_valid = np.array(x_valid), np.array(y_valid).reshape(-1, 5)

scaled_data = scaler.fit_transform(test)
x_test, y_test = [], []
for i in range(time_stamp, len(test) - 5):
    x_test.append(scaled_data[i - time_stamp:i])
    y_test.append(scaled_data[i: i + 5])
x_test, y_test = np.array(x_test), np.array(y_test).reshape(-1, 5)


def create_model():
    model = keras.Sequential()
    model.add(keras.layers.LSTM(64, return_sequences=True, input_shape=(x_train.shape[1:])))
    model.add(keras.layers.LSTM(64, return_sequences=True))
    model.add(keras.layers.LSTM(32))
    model.add(keras.layers.Dropout(0.1))
    model.add(keras.layers.Dense(5))
    model.compile(optimizer='adam', loss='mae', metrics=['accuracy'])

    return model


learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, factor=0.7,
                                                            min_lr=0.000000005)

lstm_model = create_model()

history = lstm_model.fit(x_train,y_train,batch_size=128,epochs=70,
                         validation_data=(x_valid,y_valid),
                         callbacks=[learning_rate_reduction])

plt.plot(history.history['loss'],label='training loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.show()

closing_price = lstm_model.predict(x_test)
lstm_model.evaluate(x_test)
scaler.fit_transform(pd.DataFrame(valid['开盘价(元/点)_OpPr'].values))

closing_price = scaler.inverse_transform(closing_price.reshape(-1,5)[:,0].reshape(1,-1)) #只取第一列
y_test = scaler.inverse_transform(y_test.reshape(-1,5)[:,0].reshape(1,-1))

rms = np.sqrt(np.mean(np.power((y_test[0:1,5:] - closing_price[0:1,5:] ), 2)))
print(rms)
print(closing_price.shape)
print(y_test.shape)


plt.figure(figsize=(16,8))
dict_data = {
    'Predictions': closing_price.reshape(1,-1)[0],
    '开盘价(元/点)_OpPr': y_test[0]
}
data_pd = pd.DataFrame(dict_data)
plt.plot(data_pd[['开盘价(元/点)_OpPr']],linewidth=3,alpha=0.8)
plt.plot(data_pd[['Predictions']],linewidth=1.2)
plt.show()