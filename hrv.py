CFG_lfmin = 0.04   # low frequency
CFG_lfmax = 0.15
CFG_hfmin = 0.15   # high frequency
CFG_hfmax = 0.40
CFG_interpolate_freq = 4  # Hz

import numpy as np
import scipy.signal as signal
from scipy import interpolate
import matplotlib.pyplot as plt
import peakutils


def power(spec, freq, fmin, fmax):
    band = [spec[i] for i in range(len(spec)) if freq[i] >= fmin and freq[i]<fmax]
    powerinband = np.sum(band)/(2*len(spec)**2)
    return powerinband

raw = np.genfromtxt('data.txt', delimiter=',')
t = raw[:, 0]
ppg = raw[:, 1]
ppgD = signal.detrend(ppg)
indP = peakutils.indexes(ppgD, thres=0.1, min_dist=128)
IBI = np.diff(t[indP]) * 1000

tStart = t[indP[1]]
time_axis = t[indP[1:]] - tStart  # time in ms, starts at 0
step = 1.0/CFG_interpolate_freq
interp_IBI = interpolate.interp1d(time_axis, IBI)

N = int(time_axis[-1]) - 60 + 1
tHRV = np.arange(N) + 60 + tStart
HR = np.zeros((N, 1))
LF = np.zeros((N, 1))
HF = np.zeros((N, 1))
LFHF = np.zeros((N, 1))
for x in xrange(N):
    time_axis_interp = np.arange(x, x + 60, step)
    IBI_interp = interp_IBI(time_axis_interp)
    spec_tmp = np.absolute(np.fft.fft(IBI_interp))**2
    spec = spec_tmp[0:(len(spec_tmp)/2)]  # Only positive half of spectrum
    freqs = np.linspace(start=0, stop=CFG_interpolate_freq/2, num=len(spec), endpoint=True)
    HR[x] = 60/np.mean(IBI_interp)*1000
    LF[x] = power(spec, freqs, CFG_lfmin, CFG_lfmax)
    HF[x] = power(spec, freqs, CFG_hfmin, CFG_hfmax)
    LFHF[x] = LF[x]/HF[x]

plt.subplot(2, 2, 1)
plt.plot(t, ppg)
plt.plot(t[indP], ppg[indP], 'r+', ms=5, mew=2)
plt.title('BVP')
plt.subplot(2, 2, 2)
plt.plot(t[indP[1:]], IBI)
plt.title('IBI (ms)')
plt.subplot(2, 2, 3)
plt.plot(tHRV, HR)
plt.title('Heart Rate')
plt.xlabel('Time (s)')
plt.subplot(2, 2, 4)
plt.plot(tHRV, LFHF)
plt.title('LF / HF')
plt.xlabel('Time (s)')
plt.show()