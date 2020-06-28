import pandas as pd
import numpy as np
import datetime


def load_data(datafile):
    # datafile = '../data/Challenge_9_data.csv'
    # column_names = ['Id',
    #                 'ProjectNumber',
    #                 'ProductLineNumber',
    #                 'ACTIVITY_STATUS',
    #                 'DepartmentNumber',
    #                 'ActivityTypeNumber',
    #                 'Code',
    #                 'ClassNumber',
    #                 'Baseline Start Date',
    #                 'Baseline Finish Date',
    #                 'Planned Duration',
    #                 'Forecast Start Date',
    #                 'Forecast Finish Date',
    #                 'Forecast Duration',
    #                 'Duration Variance']

    dateparse = lambda x: datetime.datetime.strptime(x, '%d/%m/%Y %H:%M')
    raw_dataset = pd.read_csv(datafile,
                              sep=",",
                              skipinitialspace=True,
                              dtype={'Id': np.int32,
                                     'ProjectNumber': np.int32,
                                     'ProductLineNumber': np.int32,
                                     'ACTIVITY_STATUS': str,
                                     'DepartmentNumber': np.int32,
                                     'ActivityTypeNumber': np.int32,
                                     'Code': str,
                                     'ClassNumber': np.int32,
                                     'Planned Duration': np.int32,
                                     'Forecast Duration': np.int32,
                                     'Duration Variance': np.int32},
                              parse_dates=['Baseline Start Date',
                                           'Baseline Finish Date',
                                           'Forecast Start Date',
                                           'Forecast Finish Date'],
                              date_parser=dateparse
                              )

    dataset = raw_dataset.copy()
    return dataset, raw_dataset


def get_quarter(x):
    return (x - 1) // 3 + 1


def main():
    original_datafile = '../data/Challenge_9_data.csv'
    pm_datafile = '../data/Challenge 9 data PM.csv'

    # loading original data
    original_dataset, original_raw_dataset = load_data(original_datafile)

    # additional data by Plamen
    pmData = pd.read_csv(pm_datafile, sep=",", skipinitialspace=True, na_values=np.nan)
    pmData.fillna(0, inplace=True)

    dataset = pd.merge(original_dataset, pmData, on='Id')
    raw_dataset = pd.merge(original_raw_dataset, pmData, on='Id')

    ###########################
    # new features
    ###########################
    # start quarter
    dataset['Baseline Start Date'] = pd.to_datetime(raw_dataset['Baseline Start Date'])
    dataset['Baseline Quarter Start'] = get_quarter(raw_dataset['Baseline Start Date'].dt.month)
    dataset['Baseline Start Month'] = raw_dataset['Baseline Start Date'].dt.month

    # end quarter
    dataset['Baseline Finish Date'] = pd.to_datetime(raw_dataset['Baseline Finish Date'])
    dataset['Baseline Quarter Finish'] = get_quarter(raw_dataset['Baseline Finish Date'].dt.month)
    dataset['Baseline Finish Month'] = raw_dataset['Baseline Finish Date'].dt.month

    # start quarter
    dataset['Forecast Start Date'] = pd.to_datetime(raw_dataset['Forecast Start Date'])
    dataset['Forecast Quarter Start'] = get_quarter(raw_dataset['Forecast Start Date'].dt.month)
    dataset['Forecast Start Month'] = raw_dataset['Forecast Start Date'].dt.month

    # end quarter
    dataset['Forecast Finish Date'] = pd.to_datetime(raw_dataset['Forecast Finish Date'])
    dataset['Forecast Quarter Finish'] = get_quarter(raw_dataset['Forecast Finish Date'].dt.month)
    dataset['Forecast Finish Month'] = raw_dataset['Forecast Finish Date'].dt.month

    # delayed start
    dataset['Delayed Start'] = raw_dataset['Baseline Start Date'] == raw_dataset['Forecast Start Date']

    # delay generally
    dataset['Delay'] = raw_dataset['Planned Duration'] < raw_dataset['Forecast Duration']

    # relative duration variance
    dataset['Relative Duration Variance'] = np.divide(raw_dataset['Duration Variance'], raw_dataset['Planned Duration'])

    # export all
    dataset.to_csv('../data/Challenge_9_newFeatures_all.csv', index=False)

    #################################
    # filtering
    #################################
    dataset = dataset[dataset['Planned Duration'] > 0]
    dataset = dataset[dataset['Planned Duration'] < 365]
    dataset = dataset[dataset['Forecast Duration'] > 0]
    dataset = dataset[dataset['ACTIVITY_STATUS'] == 'Completed']
    dataset = dataset[dataset['Forecast Finish Date'].dt.date < datetime.date(2020, 6, 26)]
    dataset = dataset[dataset['Forecast Start Date'].dt.date < datetime.date(2020, 6, 26)]

    #################################
    # export back to csv
    #################################
    dataset.to_csv('../data/Challenge_9_newFeatures_completed.csv', index=False)

    # split by length of planned duration
    qhigh, qlow = np.percentile(dataset['Planned Duration'], [66, 33])
    dataset[dataset['Planned Duration'] >= qhigh].to_csv('../data/Challenge_9_newFeatures_completed_long.csv', index=False)
    dataset[dataset['Planned Duration'] <= qlow].to_csv('../data/Challenge_9_newFeatures_completed_short.csv', index=False)

    temp = dataset[dataset['Planned Duration'] < qhigh]
    temp[temp['Planned Duration'] > qlow].to_csv('../data/Challenge_9_newFeatures_completed_medium.csv', index=False)


if __name__ == '__main__':
    main()
