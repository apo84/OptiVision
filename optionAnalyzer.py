import math as math
from itertools import combinations
import numpy as np
import pandas as pd
from yahoo_fin import options
from yahoo_fin import stock_info as yf
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import statistics
import sklearn.metrics as metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestClassifier
import sklearn as sklearn
import datetime




#def scale_dataset(dataframe, oversample=False):
#  if not isinstance(dataframe, pd.DataFrame):
#    dataframe = pd.DataFrame(dataframe)
#  x = dataframe[dataframe.columns[:-1]].values
#  y = dataframe[dataframe.columns[-1]].values

# y = np.reshape(y, (-1, 1))

#  data = np.hstack((x, y))

#  return data, x, y

def scale_dataset(dataframe, oversample=False):
  if not isinstance(dataframe, pd.DataFrame):
    dataframe = pd.DataFrame(dataframe)
  x = dataframe[dataframe.columns[:-1]].values
  y = dataframe[dataframe.columns[-1]].values

  nan_indices = np.isnan(x).any(axis=1) | np.isnan(y)
  x = x[~nan_indices]  # Select rows without NaN in y
  y = y[~nan_indices]

  scalar = StandardScaler()
  x = scalar.fit_transform(x)

  if oversample:
    ros = RandomOverSampler()
    x, y = ros.fit_resample(x, y)

  y = np.reshape(y, (-1, 1))

  data = np.hstack((x, y))

  return data, x, y

#
# TOT
#
TOT_start = '03/19/2020'
def TOT(ticker, end_date, time_period, percent_increase, tester = False):
  start_date = '03/19/2020'
  #needs to be time_period longer than other data

  #Keys: 'open', 'high', 'low', 'close', 'adjclose', 'volume', 'ticker'
  day = yf.get_data(ticker, start_date, end_date)

  open_price = day['open']
  day.index = pd.to_datetime(day.index)
  dates = day.index

  #arrays appended upon in order to graph data
  yes_increase = []
  yes_increase_date = []
  no_increase = []
  no_increase_date = []
  tot = []
  tot_date = []
  count = 0
  total = 0
  tot_cur = 0

  for x in range(dates.size-time_period):
      start = open_price[dates[x]]#first open price in the tested block
      high = open_price[dates[x]]
      i = x
      #find highest price over period when the date is closer to the end than the time period
      if dates.size - x <= time_period:
        pass
          #finds highest open price in range
          #for z in range(dates.size - x):
              #if open_price[dates[i]] > high:
                  #high = open_price[dates[i]]
              #i += 1
          #if ((high - start) / start) * 100 >= percent_increase:
              #tot.append(True)
          #else:
              #tot.append(False)
      else:
          for z in range(time_period):
              if open_price[dates[i]] > high:
                  high = open_price[dates[i]]
              i += 1
          if((high - start) / start)*100 >= percent_increase:
              tot.append(1)
          else:
              tot.append(0)

  return tot, tot_cur




#
# RSI
#
RSI_start = '02/28/2020'
def RSI(data, time_period):
  #need 14 more days
  start_date = '02/28/2020'

  rsi = []
  close = data["close"]

  for x in range(close.size-14):
    gain_count = 0
    loss_count = 0
    avg_gain = 0
    avg_loss = 0
    current_gain = 0
    current_loss = 0
    for z in range(13):
      if (close.iloc[x+z+1] - close.iloc[x+z]) > 0:
        loss_count += 1
        avg_loss += close.iloc[x+z+1] - close.iloc[x+z]
      else:
        gain_count += 1
        avg_gain += abs(close.iloc[x+z+1] - close.iloc[x+z])
    if (loss_count == 0):
      rsi.append(100)
    elif (gain_count == 0):
      rsi.append(0)
    else:
      avg_loss /= loss_count
      avg_gain /= gain_count
      if (close.iloc[x+14] - close.iloc[x+13]) > 0:
        current_loss = close.iloc[x+14] - close.iloc[x+13]
      else:
        current_gain = abs(close.iloc[x+14] - close.iloc[x+13])
      rsi.append(100-(100/(1+((avg_gain*13)+current_gain)/(((avg_loss*13)+current_loss)))))

  rsi_cur = rsi[len(rsi)-1]
  rsi = rsi[:-time_period]
  return rsi, rsi_cur



#
# ATR
#
ATR_start = '02/28/2020'
def ATR(data, time_period):

  start_date = '02/28/2020'

  dates = data.index
  atr = []

  for x in range(dates.size-14):
    avg = 0
    for z in range(14):
      high = data["high"][dates[x+1]]
      low = data["low"][dates[x+1]]
      prev_close = data["close"][dates[x]]
      methodOne = abs(high-low)
      methodTwo = abs(high-prev_close)
      methodThree = abs(low-prev_close)
      avg += max(methodOne, methodTwo, methodThree)
    avg /= 14
    atr.append(avg)

  atr_cur = atr[len(atr)-1]
  atr = atr[:-time_period]

  return atr, atr_cur




#
# OBV
#
OBV_start = '02/28/2020'
def OBV(data, time_period):

  start_date = '02/28/2020'

  price = data['close']
  dates = data.index
  obv = []

  for x in range(dates.size-14):
    cur = price.iloc[x+14]
    low = price.iloc[x]
    high = price.iloc[x]
    for z in range(14):
      if price.iloc[x+z+1] < low:
        low = price.iloc[x+z+1]
      if price.iloc[x+z+1] > high:
        high = price.iloc[x+z+1]
    cur_obv = 100*abs((cur-low)/(high-low))
    obv.append(cur_obv)

  obv_cur = obv[len(obv)-1]
  obv = obv[:-time_period]

  return obv, obv_cur





#
# PSAR
#
PSAR_start = '03/18/2020'
def PSAR(data, time_period):

  start_date = '03/18/2020'

  high = data['high']
  low = data['low']
  close = data['close']
  dates = data.index
  ep = data['low'].iloc[0]
  ep_arr = []
  ep_arr.append(ep)
  acc = 0.02
  inc = 0.02
  p_sar = data['high'].iloc[0]
  trend = 0
  initial_psar = 0
  psar = []
  psar.append(trend)

  for x in range(dates.size-2):
    sub = (p_sar-ep)*acc
    if trend == 0:
      initial_psar = max(p_sar - sub, high.iloc[x + 1], high.iloc[x])
    else:
      initial_psar = min(p_sar - sub, low.iloc[x + 1], low.iloc[x])
    if(trend==0 and high.iloc[x+2]<initial_psar):
      p_sar = initial_psar
    elif(trend==1 and low.iloc[x+2]>initial_psar):
      p_sar = initial_psar
    elif(trend==0 and high.iloc[x+2]>=initial_psar):
      p_sar = ep
    elif(trend==1 and low.iloc[x+2]<=initial_psar):
      p_sar = ep
    if(p_sar > close.iloc[x+2]):
      trend=0
    else:
      trend=1
    psar.append(trend)
    if(trend==0):
      ep = min(ep,low.iloc[x+2])
    else:
      ep = max(ep,high.iloc[x+2])
    ep_arr.append(ep)
    if(psar[x] == trend and ep_arr[x] != ep and acc<0.2):
      acc = acc+inc
    if(psar[x] == trend and ep_arr[x] == ep):
      acc = acc
    if(psar[x] != trend):
      acc = inc

  psar_cur = psar[len(psar)-1]
  psar = psar[:-time_period]

  return psar, psar_cur




#
# VMA
#
VMA_start = '03/12/2020'
def VMA(data, time_period):
  # Needs to be five days longer
  start_date = '03/12/2020'

  vol = data['volume']
  dates = data.index
  vma = []

  for x in range(dates.size-5):
    total = 0
    for z in range(5):
      total += vol.iloc[x+z]
    vma.append(total/5)

  vma_cur = vma[len(vma)-1]
  vma = vma[:-time_period]

  return vma, vma_cur



#
# Bollinger Bands
#
BB_start = '02/20/2020'
def BB(data, time_period):
  # Needs to be twenty days longer
  start_date = '02/20/2020'

  price = data['close']
  dates = data.index
  cur_mva = []
  bb = []

  for x in range(dates.size-20):
    total = 0
    for z in range(20):
      cur_mva.append(price.iloc[x+z])
    sd = statistics.stdev(cur_mva)
    bb.append(sd*2)
    cur_mva = []

  bb_cur = bb[len(bb)-1]
  bb = bb[:-time_period]

  return bb, bb_cur


#
# MACD
#
MACD_start = '01/31/2020'
def MACD(data, time_period):
    # Needs to be twenty days longer
    start_date = '01/31/2020'

    price = data['close']
    dates = data.index

    sma_twelve = 0
    mult_twelve = 2/13
    for z in range(12):
        sma_twelve += price.iloc[z]
    sma_twelve /= 12
    ema_twelve = []
    ema_twelve.append(sma_twelve)

    sma_twentysix = 0
    mult_twentysix = 2/27
    for z in range(26):
        sma_twentysix += price.iloc[z]
    sma_twentysix /= 26
    ema_twentysix = []
    ema_twentysix.append(sma_twentysix)


    for x in range(dates.size-12):
        ema_twelve.append(price.iloc[x+1]*mult_twelve + ema_twelve[x]*(1-mult_twelve))
    for x in range(dates.size-26):
        ema_twentysix.append(price.iloc[x+1]*mult_twentysix + ema_twentysix[x]*(1-mult_twentysix))

    difference = []
    for x in range(len(ema_twentysix)):
        difference.append(ema_twelve[x+14]-ema_twentysix[x])

    signal = []
    sig = 0
    for x in range(9):
        sig += difference[x]
    sig /= 9
    signal.append(sig)
    for x in range(len(difference)-9):
        sig = (difference[x+9] - signal[x])*0.25 + signal[x]
        signal.append(sig)

    hist = []
    for x in range(len(signal)):
        hist.append(difference[x+8]-signal[x])

    hist_cur = hist[len(hist)-1]
    hist = hist[:-time_period]

    return hist, hist_cur




def combination_analysis(ticker, time_period, percent_increase):
  pred_comb_list = []
  comb_list = []
  seven = [0, 1, 2, 3, 4, 5, 6]
  date = datetime.date.today()

  for i in range(5):
    knn_max_prec = 0
    knn_cur_pred = 0
    nb_max_prec = 0
    nb_cur_pred = 0
    svm_max_prec = 0
    svm_cur_pred = 0
    knn_pred_count = []
    nb_pred_count = []
    svm_pred_count = []
    knn_prec_count = []
    nb_prec_count = []
    svm_prec_count = []
    num_buys = 0


    tot, tot_cur = TOT(ticker, date, time_period, percent_increase)
    rsi, rsi_cur = RSI(yf.get_data(ticker, RSI_start, date), time_period)
    atr, atr_cur = ATR(yf.get_data(ticker, ATR_start, date), time_period)
    obv, obv_cur = OBV(yf.get_data(ticker, OBV_start, date), time_period)
    psar, psar_cur = PSAR(yf.get_data(ticker, PSAR_start, date), time_period)
    vma, vma_cur = VMA(yf.get_data(ticker, VMA_start, date), time_period)
    bb, bb_cur = BB(yf.get_data(ticker, BB_start, date), time_period)
    hist, hist_cur = MACD(yf.get_data(ticker, MACD_start, date), time_period)

    tot = pd.DataFrame(tot, columns=['tot'])
    rsi = pd.DataFrame(rsi, columns = ['rsi'])
    atr = pd.DataFrame(atr, columns=['atr'])
    obv = pd.DataFrame(obv, columns=['obv'])
    psar = pd.DataFrame(psar, columns=['psar'])
    vma = pd.DataFrame(vma, columns=['vma'])
    bb = pd.DataFrame(bb, columns=['bb'])
    macd = pd.DataFrame(hist, columns=['macd'])

    tot_cur = pd.DataFrame([tot_cur], columns=['tot'], dtype=float)
    rsi_cur = pd.DataFrame([rsi_cur], columns=['rsi'], dtype=float)
    atr_cur = pd.DataFrame([atr_cur], columns=['atr'], dtype=float)
    obv_cur = pd.DataFrame([obv_cur], columns=['obv'], dtype=float)
    psar_cur = pd.DataFrame([psar_cur], columns=['psar'], dtype=float)
    vma_cur = pd.DataFrame([vma_cur], columns=['vma'], dtype=float)
    bb_cur = pd.DataFrame([bb_cur], columns=['bb'], dtype=float)
    macd_cur = pd.DataFrame([hist_cur], columns=['macd'], dtype=float)

    comb = [rsi, atr, obv, psar, vma, bb, macd]
    pred_comb = [rsi_cur, atr_cur, obv_cur, psar_cur, vma_cur, bb_cur, macd_cur]

    for k in range(6):
      cur_comb_intial = combinations(seven, k+2)
      cur_comb = list(map(list, cur_comb_intial))
      for x in range(len(cur_comb)): #iterating through the combinations
        temp = cur_comb[x]
        comb_list = []
        pred_comb_list = []
        for z in range(len(temp)): #iterating through the current combination of numbers
          comb_list.append(comb[temp[z]]) #adding the corresponding indicator to the comb_list
          pred_comb_list.append(pred_comb[temp[z]]) #adding the corresponding indicator to the pred_comb_list
        comb_list.append(tot)
        pred_comb_list.append(tot_cur)
        df = pd.concat(comb_list, axis=1)
        cur_df = pd.concat(pred_comb_list, axis=1)
        train, test = sklearn.model_selection.train_test_split(df, test_size=0.3, train_size=0.7)
        train, x_train, y_train = scale_dataset(train, oversample=True)
        test, x_test, y_test = scale_dataset(test, oversample=False)
        cur_data, x_cur, y_cur = scale_dataset(cur_df, oversample=False)

        knn_model = KNeighborsClassifier(n_neighbors = 1)
        knn_model.fit(x_train, y_train.ravel())
        pred = knn_model.predict(x_test)
        if(metrics.precision_score(y_test, pred, zero_division=0) > knn_max_prec):
          knn_max_prec = metrics.precision_score(y_test, pred, zero_division=0)
          knn_max_num = temp
          knn_cur_pred = knn_model.predict(x_cur)

        svm_model = SVC()
        svm_model = svm_model.fit(x_train, y_train.ravel())
        pred = svm_model.predict(x_test)
        if(metrics.precision_score(y_test, pred, zero_division=0) > svm_max_prec):
          svm_max_prec = metrics.precision_score(y_test, pred, zero_division=0)
          svm_max_num = temp
          svm_cur_pred = svm_model.predict(x_cur)

        nb_model = GaussianNB()
        nb_model = nb_model.fit(x_train, y_train.ravel())
        pred = nb_model.predict(x_test)
        if(metrics.precision_score(y_test, pred, zero_division=0) > nb_max_prec):
          nb_max_prec = metrics.precision_score(y_test, pred, zero_division=0)
          nb_max_num = temp
          nb_cur_pred = nb_model.predict(x_cur)

    knn_pred_count.append(knn_cur_pred)
    knn_prec_count.append(knn_max_prec)
    nb_pred_count.append(nb_cur_pred)
    nb_prec_count.append(nb_max_prec)
    svm_pred_count.append(svm_cur_pred)
    svm_prec_count.append(svm_max_prec)

    if(svm_max_prec*svm_cur_pred + nb_max_prec*nb_cur_pred + knn_max_prec*knn_cur_pred > 1.6):
        num_buys += 1

  return num_buys, knn_pred_count, knn_prec_count, nb_pred_count, nb_prec_count, svm_pred_count, svm_prec_count