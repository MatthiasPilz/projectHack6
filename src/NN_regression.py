import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling


def build_model(input_size=1):
    model = keras.Sequential([
                            layers.Dense(64, activation='relu', input_shape=[input_size]),
                            layers.Dense(64, activation='relu'),
                            layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def main():
    in_data = pd.read_csv('../data/Challenge_9_newFeatures_completed_medium.csv',
                          sep=',',
                          na_values=np.nan,
                          dtype={'Id': np.int32,
                                 'ProjectNumber': 'category',
                                 'ProductLineNumber': 'category',
                                 'ACTIVITY_STATUS': str,
                                 'DepartmentNumber': 'category',
                                 'ActivityTypeNumber': 'category',
                                 'Code': 'category',
                                 'ClassNumber': 'category',
                                 'Planned Duration': np.int32,
                                 'Forecast Duration': np.int32,
                                 'Duration Variance': np.int32,
                                 'PM_Code_l1_past_mean': np.float64,
                                 'PM_Code_l2_past_mean': np.float64,
                                 'PM_Code_l3_past_mean': np.float64,
                                 'PM_Code_l4_past_mean': np.float64,
                                 'Baseline Quarter Start': 'category',
                                 'Baseline Start Month': 'category',
                                 'Baseline Quarter Finish': 'category',
                                 'Baseline Finish Month': 'category',
                                 'Forecast Quarter Start': 'category',
                                 'Forecast Start Month': 'category',
                                 'Forecast Quarter Finish': 'category',
                                 'Forecast Finish Month': 'category',
                                 'Delayed Start': 'bool',
                                 'Delay': 'bool',
                                 'Relative Duration Variance': np.float64,
                                 },
                          )
    print(in_data.head())

    temp = in_data.drop('ACTIVITY_STATUS', 1)
    temp = temp.drop('Id', 1)
    temp = temp.drop('Baseline Start Date', 1)
    temp = temp.drop('Baseline Finish Date', 1)
    temp = temp.drop('Forecast Start Date', 1)
    temp = temp.drop('Forecast Finish Date', 1)
    temp = pd.get_dummies(temp, prefix='', prefix_sep='')

    # X = temp.to_numpy()
    # y = np.array(in_data['Forecast Duration'])

    train_dataset = temp.sample(frac=0.8, random_state=0)
    test_dataset = temp.drop(train_dataset.index)

    # train_stats = train_dataset.describe()
    # train_stats.pop('Forecast Duration')
    # train_stats = train_stats.transpose()

    train_labels = train_dataset.pop('Forecast Duration')
    test_labels = test_dataset.pop('Forecast Duration')

    model = build_model(input_size = len(train_dataset.keys()))

    EPOCHS = 1000
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    history = model.fit(train_dataset,
                        train_labels,
                        epochs=EPOCHS,
                        validation_split = 0.2,
                        verbose=0,
                        callbacks=[tfdocs.modeling.EpochDots(), early_stop])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    # hist.tail()

    plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
    plotter.plot({'Basic': history}, metric = "mean_absolute_error")
    plt.ylim([0, 10])
    plt.ylabel('MAE [Forecast Duration]')
    plt.show()

    loss, mae, mse = model.evaluate(test_dataset, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} Forecast Duration".format(mae))

    test_predictions = model.predict(test_dataset).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [Forecast Duration]')
    plt.ylabel('Predictions [Forecast Duration]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()


if __name__ == '__main__':
    main()