from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import callbacks
import datetime


class GruPredictorArray:
    def makeModelAndFit(self, floatData, X, y, X_val, y_val, numEpochs, batchSize):
        model = Sequential()
        model.add(layers.GRU(4, input_shape=(None, floatData.shape[-1])))
        model.add(layers.Dense(floatData.shape[-1]))

        now = datetime.datetime.now()
        timeStr = now.isoformat().replace(':', '-')

        callbacksList = [
            callbacks.ModelCheckpoint(filepath=f'saved-models/model_{timeStr}.h5', monitor='val_loss', save_best_only=True)
        ]

        optimizer = RMSprop()
        model.compile(optimizer=optimizer, loss='mae', metrics=['acc'])
        print(model.summary())

        history = model.fit(X, y, epochs=numEpochs, batch_size=batchSize, callbacks=callbacksList, validation_data=(X_val, y_val))

        return history
