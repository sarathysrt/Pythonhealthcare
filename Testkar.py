# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 17:56:24 2018

@author: V750679
"""
import numpy as np
import matplotlib.pyplot as plt
#from time import gmtime, strftime
from scipy.signal import butter, lfilter
import pandas as pd
import bwr1


def bandpass_filter( data, lowcut, highcut, signal_freq, filter_order):
    nyquist_freq = 0.5 * signal_freq
    low = lowcut / nyquist_freq
    high = highcut / nyquist_freq
    b, a = butter(filter_order, [low, high], btype="band")
    y = lfilter(b, a, data)
    return y
def Find_peak(data1,rolling=40,spacing=1.5):
    data=pd.DataFrame(data1)
    data.columns=['Record']
    mx=np.max(data.Record)/4
    #ak=(np.max(data.Record)+np.min(data.Record))/2
    dataqw = [mx for i in data.Record]
    window = []
    peaklist = []
    listpos = 0 #We use a counter to move over the different data columns
    for datapoint in data.Record:
        rollingmean = dataqw[listpos] #Get local mean
        if (datapoint < rollingmean) and (len(window) < 1): #If no detectable R-complex activity -> do nothing
            listpos += 1
        elif (datapoint >= rollingmean): #If signal comes above local mean, mark ROI
            window.append(datapoint)
            listpos += 1
        else: #If signal drops below local mean -> determine highest point
            if len(str(rollingmean))==0:
                print(1)
            else:
                maximum = max(window)
                beatposition = listpos - len(window) + (window.index(maximum)) #Notate the position of the point on the X-axis
                peaklist.append(beatposition) #Add detected peak to list
                window = [] #Clear marked ROI
                listpos += 1
    return np.sort(peaklist)

def Q_peak(data,q_index):      
    i_point=0   
    q_list=[]
    for Qpoint in q_index:
        Int_measure_points = data[i_point:Qpoint]
            #Int_measure_points = Int_meas[0:195]
        Len_size=len(Int_measure_points)
        pos=0
        gate_enable=0
        for s in range(Len_size):
            t1=Int_measure_points[Len_size-(1+s)]
            t2=Int_measure_points[Len_size-(2+s)]
            if(t1>=t2):                    
                pos+=1
            else:
                if(gate_enable==0):
                    q_list.append((Len_size-(s+1)+i_point))
                    gate_enable=2
        i_point=Qpoint
    return np.sort(q_list)
def P_peak(data,Q_list):
    p_frequency= round(len(data)*0.02)
    P_list=[]
    #P_list_trail=[x-p_frequency for x in Q_list]
    P_list_trail=[]
    for i in Q_list:
        tmp=i-p_frequency
        if(tmp<=0):
            P_list_trail.append(0)
        else:
            P_list_trail.append(tmp)
    for sp in range(len(P_list_trail)):
        sk=data[P_list_trail[sp-1]:Q_list[sp-1]]
        Max_p_val=np.max(sk)
        P_max_index = np.where(sk==Max_p_val)
        P_list.append(np.min(P_list_trail[sp-1])+np.min(P_max_index[0]))

    return np.sort(P_list)

def min_max_correction(data,List1,List2,tp):
    Corr_data=[] 
    len_list=len(List1)
    for dp in range(len_list):
        sk=data[List1[dp]:List2[dp]]
        if(tp=='max'):
            Max_p_val=np.max(sk)
        else:
            Max_p_val=np.min(sk)
            
        P_max_index = np.where(sk==Max_p_val)
        Corr_data.append(np.min(List1[dp])+np.min(P_max_index[0])) 
        
    return np.sort(Corr_data)

def Data_Correction(data,image_type,baseline):
    if(image_type=='inverted'):
        max_hr=(np.max(data))
        ecg_measurements = [max_hr-x for x in data]
    else:
        ecg_measurements = data
    print('*')
    print(type(ecg_measurements))
    (baseline, ecg_measurements1) = bwr1.bwr(ecg_measurements)
    x1=min(ecg_measurements1)*-1
    Data = [x1+x for x in ecg_measurements1]
    return Data

def Peak_correction(data,pt):
    Peak_correction=[]
    for Dpoint in data:
        if(Dpoint>=0 and pt=='S'):
            Peak_correction.append(0)
        elif(Dpoint<=0 and pt=='R'):
            Peak_correction.append(0)
        else:
            Peak_correction.append(Dpoint)
    Peak_correction=np.asarray(Peak_correction)
    Peak_correction=Peak_correction**2
    return Peak_correction


integration_window = 15  # Change proportionally when adjusting frequency (in samples).
findpeaks_limit = 0.35
findpeaks_spacing = 200  # Change proportionally when adjusting frequency (in samples).
signal_frequency = 100  # Set ECG device frequency in samples per second here.

ecg_data_raw = np.loadtxt("kar2_org1.csv", skiprows=2, delimiter=',',usecols=[1])
ecg_measurements=Data_Correction(ecg_data_raw,image_type='inverted',baseline='Yes')

# step 1: Measurements filtering - 0-15 Hz band pass filter.
filtered_ecg_measurements = bandpass_filter(ecg_measurements, lowcut=0.0,highcut=15.0, signal_freq=signal_frequency,filter_order=1)
filtered_ecg_measurements[:5] = filtered_ecg_measurements[5]

#step 2: Derivative - provides QRS slope information.
differentiated_ecg_measurements = np.ediff1d(filtered_ecg_measurements)



# Peak points

data=ecg_measurements
data_len=len(data)
pps=round(data_len/10)
Buffer_frequency=round(data_len*0.02)

#data=filtered_ecg_measurements

S_list = Find_peak(Peak_correction(differentiated_ecg_measurements,pt='S'))
R_list = Find_peak(Peak_correction(differentiated_ecg_measurements,pt='R'))
Q_list = Q_peak(data,R_list)
trial_S_list= S_list+Buffer_frequency#round(data_len*0.002)
trial_P_list= Q_list-Buffer_frequency

trial_T_list= S_list+Buffer_frequency
P_list = P_peak(data,Q_list)
#P_list = min_max_correction(ecg_measurements,Q_list,trial_P_list,'max')
R_list = min_max_correction(data,P_list,S_list,'max')
Q_list = min_max_correction(data,P_list,R_list,'min')
S_list = min_max_correction(data,S_list,trial_S_list,'min')

#RR_Mean = Diff_points(ecg_measurements,R_list,R_list)
RR_Mean= round(np.mean([R_list[i+1]-R_list[i] for i in range(len(R_list)-1)]))
QRS_Mean= round(np.mean([S_list[i]-Q_list[i] for i in range(len(Q_list))]))

print(QRS_Mean/pps)
bpm=round(60/(RR_Mean/(round(data_len*0.1))))
print(RR_Mean)
print(bpm)

P_Point = [data[i] for i in P_list]
R_Point = [data[i] for i in R_list]
S_Point = [data[i] for i in S_list]
Q_Point = [data[i] for i in Q_list]
trial_S_Point = [data[i] for i in trial_S_list]
trial_T_Point = [data[i] for i in trial_T_list]
print(P_Point[0])
ds=[]
k=0
for i in range(data_len):
    if(i>S_list[k]):
        if(k<len(S_list)-1):
            k+=1
    print(k,len(S_list))
    if(i>=P_list[k] and i<=S_list[k]):
        ds.append(data[i]/2)
    else:
        ds.append(data[i]**2)

plt.figure(figsize=(30, 5))
#plt.plot(ecg_measurements[:int(S_list[1])+1])
#plt.plot(data,color='gray')
#plt.plot(range(len(data)),data)#,marker='.')
plt.plot(ds)
plt.scatter(S_list,S_Point,color='red')
plt.scatter(R_list,R_Point,color='brown')
plt.scatter(Q_list,Q_Point,color='black')
plt.scatter(P_list,P_Point,color='green')
#plt.scatter(trial_S_list,trial_S_Point,color='orange')
#plt.scatter(trial_S_list,trial_S_Point)
plt.scatter(trial_T_list,trial_T_Point)
plt.grid()
plt.show()
plt.close()
