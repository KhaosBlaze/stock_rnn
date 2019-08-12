import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("stock", help="Stock in Question", type=str)
vars = parser.parse_args()
stock = vars.stock

dataset = pd.read_csv('stocks/' + stock + '.csv')
dataset = dataset.reindex(index=dataset.index[::-1])
dataset = dataset.reset_index()
dataset = dataset.drop(columns=['index'])

output = []
for i in range(46, len(dataset)):
    if dataset['close'][i] > dataset['open'][i]:
        output.append(1)
    else:
        output.append(0)

y = np.array(output).astype(int)

np.savetxt(stock+'.out', y, delimiter=',')
