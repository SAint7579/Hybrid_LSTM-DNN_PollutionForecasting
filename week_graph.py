import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
from numpy import random
#Label Encoder
lbe = LabelEncoder()

dataset=pd.read_csv('pollution.csv',index_col='date')
dataset=dataset[-24*7*12:-24*7*11]
dataset['wnd_dir'] = lbe.fit_transform(dataset['wnd_dir'])
dataset=dataset.drop('pollution',1)
dataset=dataset.drop('month',1)
# dataset=pd.DataFrame(MinMaxScaler().fit_transform(dataset))
# dataset.index.name = 'date'
# dataset.columns = ['dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
for i in dataset:
	dataset[i].plot(figsize=(16,9), color=random.rand(1,3))
	plt.xlabel('Time')
	plt.ylabel(i)
	plt.show()