from glob import glob
import numpy as np
import os
import gzip
from tqdm import tqdm
import pathlib
import datetime
from scipy.interpolate import griddata

# 환경변수 기반 사용자 설정 경로 및 변수
rad_HSP_path = os.getenv('RAD_HSP_PATH')
radtype = os.getenv('RAD_TYPE')

initial_time = os.getenv('INITIAL_TIME')
close_time = os.getenv('CLOSE_TIME')

# 시간 범위 계산
time_range = int((datetime.datetime.strptime(close_time, '%Y%m%d%H%M')
                  - datetime.datetime.strptime(initial_time, '%Y%m%d%H%M')) / datetime.timedelta(minutes=10))

savepath = os.getenv('SAVE_PATH')
os.makedirs(savepath, exist_ok=True)
missingdata = []

# 원본 데이터 크기
original_shape = (1200, 1100)

# 목표 Grid 데이터 크기
grid_shape = (300, 275)

# 원본 좌표 생성
y = np.linspace(0, original_shape[0]-1, original_shape[0])
x = np.linspace(0, original_shape[1]-1, original_shape[1])
xv, yv = np.meshgrid(x, y)

# 목표 좌표 생성
grid_y = np.linspace(0, original_shape[0]-1, grid_shape[0])
grid_x = np.linspace(0, original_shape[1]-1, grid_shape[1])
grid_xv, grid_yv = np.meshgrid(grid_x, grid_y)

for i in tqdm(range(time_range)):
    load_path = os.path.join(rad_HSP_path, 'RDR_CMP_HSP_PUB_' + initial_time + '.bin.gz')
    try:
        with gzip.open(load_path, 'rb') as f:
            radardata = np.frombuffer(f.read(), dtype=np.int16)

        radardata = radardata[512:]
        radardata = radardata.reshape(2881, 2305)
        radardata = np.flip(radardata, axis=0)

        radardata = radardata[:2880, 1:]
        radardata = radardata[1200 - 220:1200 + 980, 1121 - 328:1121 + 772]

        # Griddata로 interpolation
        points = np.vstack((xv.flatten(), yv.flatten())).T
        values = radardata.flatten()

        radardata_grid = griddata(points, values, (grid_xv, grid_yv), method='linear')

        radardata_grid = radardata_grid / 100
        radardata_grid = np.where(radardata_grid < 0, 0, radardata_grid)

        input_path_new = pathlib.PureWindowsPath(pathlib.PureWindowsPath(load_path).stem).stem

        np.save(os.path.join(savepath, input_path_new), radardata_grid)

        loadtimeToDate = datetime.datetime.strptime(initial_time, '%Y%m%d%H%M') + datetime.timedelta(minutes=10)
        initial_time = loadtimeToDate.strftime("%Y%m%d%H%M")
    except Exception as e:
        print(f"{initial_time} does not exist: {e}")
        missingdata.append(load_path)
        loadtimeToDate = datetime.datetime.strptime(initial_time, '%Y%m%d%H%M') + datetime.timedelta(minutes=10)
        initial_time = loadtimeToDate.strftime("%Y%m%d%H%M")

print(missingdata)
