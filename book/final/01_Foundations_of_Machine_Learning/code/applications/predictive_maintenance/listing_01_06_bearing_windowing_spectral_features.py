"""Listing 1.6 — Windowing and spectral feature extraction for bearing vibration."""
from __future__ import annotations
import numpy as np

def extract_windows(signal,fs,window_seconds=4.0,hop_seconds=2.0):
    signal=np.asarray(signal,dtype=float)
    wl=int(round(window_seconds*fs)); hl=int(round(hop_seconds*fs))
    if wl<=0 or hl<=0 or len(signal)<wl: raise ValueError("invalid window/hop or signal too short")
    starts=range(0,len(signal)-wl+1,hl)
    return np.stack([signal[s:s+wl] for s in starts])

def spectral_features(window,fs):
    x=np.asarray(window,dtype=float)
    spectrum=np.abs(np.fft.rfft(x)); freqs=np.fft.rfftfreq(len(x),d=1.0/fs)
    denom=max(float(spectrum.sum()),1e-12)
    centroid=float((freqs*spectrum).sum()/denom)
    rms=float(np.sqrt(np.mean(x**2)))
    centered=x-x.mean(); std=max(float(x.std()),1e-12)
    kurtosis=float(np.mean(centered**4)/(std**4))
    return np.array([centroid,rms,kurtosis],dtype=float)

if __name__=="__main__":
    fs=1000; t=np.arange(0,20,1/fs)
    signal=np.sin(2*np.pi*50*t)+.25*np.sin(2*np.pi*120*t)
    W=extract_windows(signal,fs)
    F=np.vstack([spectral_features(w,fs) for w in W])
    print("windows:",W.shape,"features:",F.shape,"first:",np.round(F[0],4))
