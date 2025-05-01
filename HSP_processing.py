from glob import glob
import numpy as np
import os
import gzip
from skimage import transform
from tqdm import tqdm
from pathlib import Path
import pathlib
import datetime
import shutil

#======================================================================================================================
rad_HSP_path = 'D:\HSP/'
rad_HSP_paths = sorted(glob(rad_HSP_path))
print(rad_HSP_paths)
radtype = 'HSP'

initial_time = '202105040940'
close_time = '202305030930'

time_range = int( (datetime.datetime.strptime(close_time, '%Y%m%d%H%M')
                   - datetime.datetime.strptime(initial_time, '%Y%m%d%H%M') )/ datetime.timedelta(minutes=10))
# print(time_range)

savepath = 'D:/HSP_train/'

os.makedirs(savepath,exist_ok=True)
missingdata = []

for i in tqdm(range(time_range)):
    load_path = rad_HSP_path + 'RDR_CMP_HSP_PUB_'+ initial_time + '.bin.gz'
    # print(load_path)
    try:
        with gzip.open(load_path,'rb') as f:
            radardata = np.frombuffer(f.read(),dtype = np.int16)
        radardata = radardata[512:]
        radardata = radardata.reshape(2881, 2305)  # radar 1440km x 1152km
        radardata = np.flip(radardata, axis=0)

        # # image cropping
        # # center = 1200,1121, size 1200,1100
        radardata = radardata[:2880, 1:]
        radardata = radardata[1200 - 220:1200 + 980, 1121 - 328:1121 + 772]  # (1200,1100) 0.5km resolution
        radardata = transform.resize(radardata, (300, 275), preserve_range=True)

        input_path_new = pathlib.PureWindowsPath(pathlib.PureWindowsPath(load_path).stem).stem

        radardata = radardata / 100
        radardata = np.where(radardata < 0, 0, radardata)
        # radardata = (radardata - 75) / 75 #normalize HSP : 0~150mm

        np.save(savepath + input_path_new, radardata)

        loadtimeToDate = datetime.datetime.strptime(initial_time, '%Y%m%d%H%M') + datetime.timedelta(minutes=10)
        initial_time = loadtimeToDate.strftime("%Y%m%d%H%M")
    except:
        print(initial_time, 'does not exist')
        missingdata.append(rad_HSP_path + 'RDR_CMP_HSP_PUB_'+ initial_time + '.bin.gz')
        loadtimeToDate = datetime.datetime.strptime(initial_time, '%Y%m%d%H%M') + datetime.timedelta(minutes=10)
        initial_time = loadtimeToDate.strftime("%Y%m%d%H%M")

#======================================================================================================================
print(missingdata)
~
