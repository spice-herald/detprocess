from ipyparallel import Client

c = Client()
c.ids

dview = c[:]
dview.block = True

from detprocess.process import process_data

dview.execute('from detprocess.process import process_data')

path_to_yaml = '/full/path/to/yaml/process.yaml'
savepath = '/full/path/to/save/processed/data/'

mydict = {
    'process_data': process_data,
    'path_to_yaml': path_to_yaml,
    'savepath': savepath,
}

dview.push(mydict)

from glob import glob
files = glob('/full/path/to/folder/to/process/*')

def wrapper(f):
    return process_data(f, path_to_yaml, savepath=savepath)

import time
time.time()

lview = c.load_balanced_view()
lview.block = True

df = lview.map(process_data, files)
time.time()

print(df.head())
