from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils

(train_img, train_label), (test_img, test_label) = fashion_mnist.load_data()

train_img = train_img.reshape(train_img.shape[0], train_img.shape[1] * train_img.shape[2])
test_img = test_img.reshape(test_img.shape[0], test_img.shape[1] * test_img.shape[2])

train_img = train_img / 255
test_img = test_img / 255

train_label = np_utils.to_categorical(train_label)
test_label = np_utils.to_categorical(test_label)

print(train_img.shape)
print(test_img.shape)
print(train_label.shape)
print(test_label.shape)

dim = train_img.shape[1]


def create_model():
    model = Sequential()
    model.add(Dense(dim, input_dim=dim, activation='relu'))
    # model.add(Dropout(0.1))
    model.add(Dense(train_label.shape[1], activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


mlp_model = create_model()

mlp_model.fit(train_img, train_label, validation_data=(test_img, test_label),
              epochs=10, batch_size=200, verbose=2)

score = mlp_model.evaluate(test_img, test_label)

print(score[1])
