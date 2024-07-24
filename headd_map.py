import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import pickle
import csv
import os
from scipy.fft import fft
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import re
import seaborn as sns
import math
import pandas as pd
import time
import PySimpleGUI as sg
import mne

sg.theme('Dark Grey 13')

layout = [[sg.Text('сюда EDF')],
          [sg.Input(), sg.FileBrowse()],
          [sg.Text('сюда ANN')],
          [sg.Input(), sg.FileBrowse()],
          [sg.Text('куда сохранить графики?')],
          [sg.Input(),sg.FolderBrowse()],
          [sg.OK(), sg.Cancel()]]

window = sg.Window('Get filename example', layout)

event, values = window.read()
window.close()


initial_EDF_R_file = mne.io.read_raw_edf(values[0])
data_from_file = initial_EDF_R_file.get_data()
chanel_names = initial_EDF_R_file.ch_names


def c_sf_spm(arr): # суммарный спектр
    SVdft = fft(arr);
    N_coef = int(SVdft.shape[0] / 2)
    N = len(arr)
    SVdft2 = SVdft[:N_coef + 1]  # %FFT - расчет КФ (комплексные)
    psdSV = 2 * ((1 / (250 * N)) * abs(SVdft2) ** 2)
    return psdSV

def razrez_spektr(eyes_close_before, wind=250, step=50):

    import PySimpleGUI as sg


    layout = [[sg.Text('A custom progress meter')],
            [sg.ProgressBar(eyes_close_before.shape[1], orientation='h', size=(20, 20), key='progressbar')],
            [sg.Cancel()]]


    window = sg.Window('Custom Progress Meter', layout)
    progress_bar = window['progressbar']
 

    
    for i in range(0,eyes_close_before.shape[1],step):
        event, values = window.read(timeout=0)
        if i+wind<=eyes_close_before.shape[1]:
            # print(i)
            data = eyes_close_before[:-4, i:i+wind]
            # print(data.shape)
            
            for j in range(data.shape[0]):
                spek_data = c_sf_spm(data[j, :])
                # spek_data = np.flip(spek_data)
                # print(spek_data.shape)
                # plt.plot(spek_data)
                # plt.show()
                
                try:
                    data_spek = np.hstack((data_spek,spek_data.reshape(-1,1)))
                except NameError:
                    data_spek = spek_data.reshape(-1,1)
            #     print(data_spek.shape)
            # break
            plt.plot(data_spek)
            plt.show()
            try:
                data3d = np.dstack((data3d, data_spek)) 
            except:
                data3d = data_spek
            del data_spek
            # print(data3d.shape)

        progress_bar.UpdateBar(i+1)
        # count+=1
        # if count == 100:
        #     break
    window.close()
    return data3d

# np.loadtxt("")
data = razrez_spektr(data_from_file)
txt_ann_file = np.genfromtxt(values[1], delimiter = ',',skip_header = 1)[:,[0]]
labels = np.genfromtxt(values[1], delimiter = ',',skip_header = 1)[:,[2]]
metki = txt_ann_file[:,0]/50
label_y = np.arange(126)
# print(label_y)
for k in range(19):
    plt.rc('ytick', labelsize=10) #fontsize of the y tick labels
    data2 = data[:,k,:]
    data3 = np.flipud(data2)
    plt.figure(figsize = (160,90))
    
    sns.heatmap(data=data3, cmap="jet", vmin =1*10**(-15), vmax = 1*10**(-11), cbar_kws={'label': 'POWER'})
    # print(np.arange(data.shape[2])/5/60)
    plt.title(chanel_names[k])
    plt.yticks(label_y,np.flipud(label_y))
    plt.xticks(metki, labels, rotation='vertical')
    plt.xlabel('time', fontsize=10)
    plt.ylabel('frequency', fontsize=10)
    # for kk in metki:
    #     plt.axvline (x=kk, color='red', linestyle='--')
    # print(values[0])
    plt.savefig(values[2]+'/'+chanel_names[k]+'_head_map.png')
    plt.close()

for k in range(19):
    for j  in range(126):

        plt.rc('ytick', labelsize=5) 
        data4 = data[j,k,:]
        # data3 = np.flipud(data2)
        plt.figure(figsize = (16,9))
        # print(data4.shape)
        plt.plot(data4)
    
        # sns.heatmap(data=data3, cmap="jet", vmin =1*10**(-15), vmax = 1*10**(-11), cbar_kws={'label': 'POWER'})
        # print(np.arange(data.shape[2])/5/60)
        plt.title(chanel_names[k]+'_frequency_'+str(j))
        # plt.yticks(label_y,np.flipud(label_y))
        plt.xticks(metki, labels, rotation='vertical')
        plt.xlabel('time', fontsize=10)
        plt.ylabel('Power', fontsize=10)
        # for kk in metki:
        #     plt.axvline (x=kk, color='red', linestyle='--')
        # print(values[0])
        plt.savefig(values[2]+'/'+chanel_names[k]+'_frequency_'+str(j)+'.png')
        plt.close()
