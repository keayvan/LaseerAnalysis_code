#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 17:27:28 2023

@author: k1
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Set the data directory
# data_directory = r'C:\Users\Pedro David García\Dropbox\OneDrive - csic.es\5. CSIC\8. Calculations\COMSOL\roughness\600 nm'

# # Change the current working directory to the data directory
# os.chdir(data_directory)

# Load data from file
data = pd.read_csv('MeshV1.csv', skiprows=5,delimiter=',')
data.columns.values[2] = 'QF'
data.columns.values[1] = 'Freq'

steps = data.shape[0]
w = data['Freq']
q = data['QF']

# w = data[:, 0]
# q = data[:, 1]
a = np.ones(steps)
reala = np.zeros(steps)
error = np.zeros(steps)

# Get system parameters
Nprpeak = 1e9
Extendrange = 0.1
df = 50000
mindf = np.min(df)
minfidx = np.argmin(w)
maxfidx = np.argmax(w)
minf = w[minfidx]
maxf = w[maxfidx]

# Extract max and min plot f
minplotf = minf - Extendrange * (maxf - minf)
maxplotf = maxf + Extendrange * (maxf - minf)
freq = np.arange(minplotf, maxplotf, (maxplotf-minplotf)/df)

field = np.zeros(freq.shape, dtype=complex)
decay = np.pi * 1.0 / (q * w)
phase = np.arccos(reala / a)

for k in range(len(w)):
    field += a[k] * np.exp(1j * phase[k]) / (1j * np.pi * (w[k] - freq) + decay[k] / 2)

I = np.abs(field) ** 2
I /= np.max(I)
# Plot the data
plt.figure()
plt.plot(freq, np.log(I))
plt.xlabel('Frequency')
plt.ylabel('log(I)')
plt.title('Spectrum')
plt.grid(True)

# Save the data
D = np.vstack((freq, I)).T
np.savetxt('Intensity-ten.txt', D, fmt='%1.6f')
plt.show()
