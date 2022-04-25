from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import callbacks
import datetime


class GruPredictorGenerator:
    def __init__(self):
        print('Model started')

    def makeModelAndFit(self, floatData, trainGen, valGen, valSteps, numEpochs, stepsPerEpoch):
        model = Sequential()
        model.add(layers.GRU(2, input_shape=(None, floatData.shape[-1])))
        # model.add(layers.Dense(floatData.shape[-1]))

        now = datetime.datetime.now()
        timeStr = now.isoformat().replace(':', '-')

        callbacksList = [
            # callbacks.EarlyStopping(monitor='val_loss', patience=1),
            callbacks.ModelCheckpoint(filepath=f'saved-models/model_{timeStr}.h5', monitor='val_loss', save_best_only=True)
        ]

        optimizer = Adam()
        model.compile(optimizer=optimizer, loss='mae', metrics=['acc'])
        print(model.summary())

        history = model.fit_generator(trainGen, steps_per_epoch=stepsPerEpoch, epochs=numEpochs, validation_data=valGen, validation_steps=valSteps, callbacks=callbacksList)

        return history
