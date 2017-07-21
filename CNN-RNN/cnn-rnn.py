import os
import sys

sys.path.append(os.pardir)

import utill


if __name__ == '__main__':
  data = utill.csv_to_main_data("../data/bitcoin_ticker.csv")
  x_data, y_data = utill.split_series_data(data, 30)

