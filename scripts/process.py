from ipyparallel import Client

c = Client()
c.ids

dview = c[:]
dview.block = True

from detprocess.process import process_data

dview.execute('from detprocess.process import process_data')

mydict = {'process_data': process_data}
dview.push(mydict)

from glob import glob
files = glob('/full/path/to/folder/to/process/*')

import time
time.time()

lview = c.load_balanced_view()
lview.block = True

df = lview.map(process_data, files)
time.time()

print(df.head())
