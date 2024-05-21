import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
import os
import csv
from scipy import stats as st
from utils import *

import torch.nn as nn
import torch.optim as optim
import torchvision
#import torchvision.datasets as datasets

# from arch.bootstrap import MovingBlockBootstrap
# from arch.bootstrap import optimal_block_length
# from arch.bootstrap import CircularBlockBootstrap
# from arch.bootstrap import StationaryBootstrap
from numpy.random import standard_normal

import numpy as np
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from torch.autograd import Variable
import math
from sklearn.manifold import TSNE
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance
from TS_DEC import *
#from TS_DEC_Linear import *
import random
import sys

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from scipy.spatial.distance import cosine as cosine_distance
from typing import Optional, List
from scipy.io import arff
from scipy.stats import entropy
from scipy.signal import find_peaks
import ruptures as rpt
from scipy.signal import savgol_filter
# from dbscan1d.core import DBSCAN1D

import pandas as pd
import numpy as np
#import stumpy
#from stumpy.floss import _cac
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib import animation
from IPython.display import HTML
import os
import re
from statsmodels.tsa.seasonal import STL

from KDEpy import FFTKDE
from sklearn.neighbors import KernelDensity
from numpy import array, linspace
import statsmodels.api as sm
from statsmodels.distributions.mixture_rvs import mixture_rvs

import warnings
warnings.filterwarnings('ignore')

orginal_ts = np.loadtxt("./data/PAMAP2.csv", delimiter=",")
labels = np.loadtxt("./data/PAMAP2_labels.csv", delimiter=",")
ground_truth = np.where(labels[:-1] != labels[1:])[0]

def ts_uncertainty_gridtest(data, ground_truth, sample_rate, n, model_settings, inference_settings, print_info=True):
    filtered_data = filter_data(data, sample_rate)
    
    all_predictions = []
    all_coverings = []
    all_f1s = []
    for i in range(n):
        residuals = data-filtered_data
        noise =  np.random.default_rng().uniform(0.0,2.0,len(data))
        signal = filtered_data + (data-filtered_data)*noise
        
        
        for model_setting in model_settings:
            for inference_setting in inference_settings:
                
                mp = stumpy.stump(signal,m=int(model_setting*sample_rate))
                cac, regime_locations = stumpy.fluss(mp[:, 1], L=sample_rate, n_regimes=1, excl_factor=1)
                predictions, _ = find_peaks(-cac + 1,height=0.25,prominence=inference_setting)
                all_predictions.append(predictions)
                
                margin = 0.01*len(signal) +  sample_rate
                covering_score = covering(ground_truth, predictions, len(signal))
                all_coverings.append(covering_score)
                f1_score = f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=False)
                all_f1s.append(f1_score)
                if print_info:
                    print('covering score:', covering_score)
                    print('f_measure score:', f1_score)
                    plt.plot(cac)
                    for regime_location in predictions:
                        plt.axvline(x=regime_location, linestyle="dashed")
                    for regime_location in ground_truth:
                        plt.axvline(x=regime_location, linestyle="dashed",color="red")
                    plt.show()
        
    all_predictions = np.array(all_predictions)
    all_coverings = np.array(all_coverings)
    all_f1s = np.array(all_f1s)
    return all_predictions, all_coverings, all_f1s

def ts_uncertainty_randomtest(data, ground_truth, sample_rate, n, model_settings, inference_settings, print_info=True):
    filtered_data = filter_data(data, sample_rate)
    
    all_predictions = []
    all_coverings = []
    all_f1s = []
    for i in range(n):
        residuals = data-filtered_data
        noise =  np.random.default_rng().uniform(0.5,1.5,len(data))
        print(noise)
        fig, axs = plt.subplots(2,sharex=True, gridspec_kw={'hspace': 0})
        axs[0].plot(data)
      
        signal = filtered_data + (data-filtered_data)*noise
        axs[1].plot(signal)
        axs[0].set_title("Orignal Time Series (Top) and Perturbed Time Series (Bottom) Comparison",size=16)
        plt.xlabel("Observation(n)",size=14)
        axs[0].set_ylabel("Amplitude",size=14)
        axs[1].set_ylabel("Amplitude",size=14)
        axs[0].set_ylim([-12.5, 7.0])
        axs[1].set_ylim([-12.5, 7.0])
        plt.show()
        fig.savefig("tscomparison.png",dpi=400) 
        sys.exit()
        i1 = random.randint(0, len(model_settings)-1)
        i2 = random.randint(0, len(inference_settings)-1)
        print(i1,i2)
        model_setting = model_settings[i1]
        inference_setting = inference_settings[i2]
        
        print(model_setting, inference_setting)        
        mp = stumpy.stump(signal,m=int(model_setting*sample_rate))
        cac, regime_locations = stumpy.fluss(mp[:, 1], L=sample_rate, n_regimes=1, excl_factor=1)
        predictions, _ = find_peaks(-cac + 1,height=0.25,prominence=inference_setting)
        all_predictions.append(predictions)

        margin = 0.01*len(signal) +  sample_rate
        covering_score = covering(ground_truth, predictions, len(signal))
        all_coverings.append(covering_score)
        f1_score = f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=False)
        all_f1s.append(f1_score)
        if print_info:
            print('covering score:', covering_score)
            print('f_measure score:', f1_score)
            plt.plot(cac)
            for regime_location in predictions:
                plt.axvline(x=regime_location, linestyle="dashed")
            for regime_location in ground_truth:
                plt.axvline(x=regime_location, linestyle="dashed",color="red")
            plt.show()

    all_predictions = np.array(all_predictions)
    all_coverings = np.array(all_coverings)
    all_f1s = np.array(all_f1s)
    return all_predictions, all_coverings, all_f1s

def ts_uncertainty_data(data, ground_truth, sample_rate, n, model_setting, inference_setting, print_info=True):
    filtered_data = filter_data(data, sample_rate)
    
    all_predictions = []
    all_coverings = []
    all_f1s = []
    for i in range(n):
        residuals = data-filtered_data
        noise =  np.random.default_rng().uniform(0.0,2.0,len(data))
        signal = filtered_data + (data-filtered_data)*noise
        
                
        mp = stumpy.stump(signal,m=int(model_setting*sample_rate))
        cac, regime_locations = stumpy.fluss(mp[:, 1], L=sample_rate, n_regimes=1, excl_factor=1)
        predictions, _ = find_peaks(-cac + 1,height=0.25,prominence=inference_setting)
        all_predictions.append(predictions)

        margin = 0.01*len(data) +  sample_rate
        covering_score = covering(ground_truth, predictions, len(data))
        all_coverings.append(covering_score)
        f1_score = f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=False)
        all_f1s.append(f1_score)
        if print_info:
            print('covering score:', covering_score)
            print('f_measure score:', f1_score)
            plt.plot(cac)
            for regime_location in predictions:
                plt.axvline(x=regime_location, linestyle="dashed")
            for regime_location in ground_truth:
                plt.axvline(x=regime_location, linestyle="dashed",color="red")
            plt.show()

    all_predictions = np.array(all_predictions)
    all_coverings = np.array(all_coverings)
    all_f1s = np.array(all_f1s)
    return all_predictions, all_coverings, all_f1s

def ts_uncertainty_model(data, ground_truth, sample_rate, model_settings, inference_setting, print_info=True):
    
    all_predictions = []
    all_coverings = []
    all_f1s = []
    for model_setting in model_settings:
        
        signal = data
                
        mp = stumpy.stump(signal,m=int(model_setting*sample_rate))
        cac, regime_locations = stumpy.fluss(mp[:, 1], L=sample_rate, n_regimes=1, excl_factor=1)
        predictions, _ = find_peaks(-cac + 1,height=0.25,prominence=inference_setting)
        all_predictions.append(predictions)

        margin = 0.01*len(data) +  sample_rate
        covering_score = covering(ground_truth, predictions, len(data))
        all_coverings.append(covering_score)
        f1_score = f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=False)
        all_f1s.append(f1_score)
        if print_info:
            print('covering score:', covering_score)
            print('f_measure score:', f1_score)
            plt.plot(cac)
            for regime_location in predictions:
                plt.axvline(x=regime_location, linestyle="dashed")
            for regime_location in ground_truth:
                plt.axvline(x=regime_location, linestyle="dashed",color="red")
            plt.show()

    all_predictions = np.array(all_predictions)
    all_coverings = np.array(all_coverings)
    all_f1s = np.array(all_f1s)
    return all_predictions, all_coverings, all_f1s

def ts_uncertainty_inference(data, ground_truth, sample_rate, model_setting, inference_settings, print_info=True):
    
    all_predictions = []
    all_coverings = []
    all_f1s = []
    for inference_setting in inference_settings:
        
        signal = data
                
        mp = stumpy.stump(signal,m=int(model_setting*sample_rate))
        cac, regime_locations = stumpy.fluss(mp[:, 1], L=sample_rate, n_regimes=1, excl_factor=1)
        predictions, _ = find_peaks(-cac + 1,height=0.25,prominence=inference_setting)
        all_predictions.append(predictions)

        margin = 0.01*len(data) +  sample_rate
        covering_score = covering(ground_truth, predictions, len(data))
        all_coverings.append(covering_score)
        f1_score = f_measure(ground_truth, predictions, margin=margin, alpha=0.5, return_PR=False)
        all_f1s.append(f1_score)
        if print_info:
            print('covering score:', covering_score)
            print('f_measure score:', f1_score)
            plt.plot(cac)
            for regime_location in predictions:
                plt.axvline(x=regime_location, linestyle="dashed")
            for regime_location in ground_truth:
                plt.axvline(x=regime_location, linestyle="dashed",color="red")
            plt.show()

    all_predictions = np.array(all_predictions)
    all_coverings = np.array(all_coverings)
    all_f1s = np.array(all_f1s)
    return all_predictions, all_coverings, all_f1s

def filter_data(data, sample_rate):
    sample_rate = len(data)//200
    mask=np.ones((1,sample_rate)) /sample_rate
    mask=mask[0,:]
    filtered_data = np.convolve(data,mask,'same')
    filtered_data = np.squeeze(filtered_data)
    
    return filtered_data



penalty = sys.argv[1]
penalty = int(penalty)
penalties=[penalty-2000,penalty-1000,penalty,penalty+1000,penalty+2000]
jumps = [0.8*100,0.9*100,100,1.1*100,1.2*100]
models = ["l1","l2","rbf"]
all_predictions = []
covering_scores = []
print(penalties)

for o in range(100):
    signal = np.copy(orginal_ts)
    sample_rate = 500
    mask=np.ones((1,sample_rate)) /sample_rate
    mask=mask[0,:]
    for i,ts in enumerate(signal):
        filtered_time_series = np.convolve(ts,mask,'same')
        noise =  np.random.default_rng().uniform(0.5,1.5,len(ts))
        signal[i] = filtered_time_series + (ts-filtered_time_series)*noise
    
    
    # change point detection
    model = "l2"  # "l2", "rbf"
    i1 = random.randint(0, len(penalties)-1)
    i2 = random.randint(0, len(jumps)-1)
    print(i1, i2)
    algo = rpt.Pelt(model=model, min_size=3000, jump=int(jumps[i2])).fit(signal.T)
    predictions = algo.predict(pen=int(penalties[i1]))
    all_predictions.append(predictions)
    activities = []
    activities.append(labels[0])
    for l in labels:
        if l != activities[-1]:
            activities.append(int(l))


    activity_names = ['Transition',
                      'Lying',
                      'Sitting',
                      'Standing',
                      'Walking',
                      'Running',
                      'Cycling',
                      'Nordic Walking',
                      '8',
                      'Watching TV',
                      'Computer Work',
                      'Car Driving',
                      'Ascending Stairs',
                      'Descending Stairs',
                      '14',
                      '15',
                      'Vacuum Cleaning',
                      'Ironing',
                      'Folding Laundry',
                      'House Cleaning',
                      'Playing Soccer',
                      '21',
                      '22',
                      '23',
                      'Rope Jumping']

    
    covering_scores.append(covering(ground_truth, predictions, len(signal[0])))
    print('covering score:',covering(ground_truth, predictions, len(signal[0])))

for ap in all_predictions:
    print(ap)

#np.savetxt('all_predictions_34000.out', all_predictions, delimiter=',',fmt='%s')
    