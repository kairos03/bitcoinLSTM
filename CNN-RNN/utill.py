# data utill
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


def csv_to_main_data(file):
  df = pd.read_csv(file)
  df = df.loc[(df['rpt_key'] == 'btc_krw')]
  df['datetime'] = pd.to_datetime(df['datetime_id'])
  df = df.loc[df['datetime'] > pd.to_datetime('2017-06-28 00:00:00')]
  df = df.reset_index(drop=True)
  df = df[['last', 'low', 'high', 'bid', 'ask', 'volume', 'diff_24h', 'diff_per_24h']]

  data = df[['last', 'low', 'high', 'bid', 'ask', 'diff_24h', 'volume', 'diff_per_24h']]

  plt.plot(data)
  plt.show()

  return data


def split_series_data(data, step):
  x_data = []
  y_data = []

  pre_last = 0
  for i in range(data.shape[0]/step):
    xt_data = data[i*step:(i+1)*step]
    x_data.append(xt_data)

    now_last = xt_data['last'].iloc[0]
    if pre_last < now_last:
      yt_data = 1
    else:
      yt_data = 0
    pre_last = now_last
    y_data.append(yt_data)

  print x_data,
  print y_data
  print len(x_data), len(y_data)
  print x_data[0].shape

  return x_data, y_data
