import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
def read_data(folder,frame):
    plt.clf()
    data = []
    for data_type in ['depth_log','rgb','seg']:
        path = os.path.join(folder,f"{data_type}_{frame}.png")
        data.append(cv2.imread(path))
    result = np.hstack(data)
    cv2.imwrite(f'{folder}/combined_{frame}.png',result)
        
folder = '25052020_000859_out_dynamic_weather'
for frame in np.arange(1200,2000,100):
    read_data(folder,frame)
