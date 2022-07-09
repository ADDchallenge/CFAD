import soundfile as sf
import librosa
import numpy as np
import os
import sys
import time
from scipy.fftpack import dct
from multiprocessing import Process
import math

def cal_time(func):
    def warpper(*args, **kwargs):
        st = time.time()
        res = func(*args, **kwargs)
        et = time.time()
        # print('[Function: {name} finished, spent time: {time:.6f}s]'.format(name=func.__name__, time=et-st))
        return res
    return warpper

@cal_time
def pre_emphasis(sig, pre_emph_coeff=0.97):
    """
    perform preemphasis on the input signal.
    Args:
        sig   (array) : signal to filter.
        coeff (float) : preemphasis coefficient. 0 is no filter, default is 0.95.
    Returns:
        the filtered signal.
    """
    return np.append(sig[0], sig[1:] - pre_emph_coeff * sig[:-1])

@cal_time
def stride_trick(a, stride_length, stride_step):
    """
    apply framing using the stride trick from numpy.
    Args:
        a (array) : signal array.
        stride_length (int) : length of the stride.
        stride_step (int) : stride step.
    Returns:
        blocked/framed array.
    """
    #nrows = (math.ceil((a.size - stride_length) / stride_step)) + 1
    nrows = ((a.size - stride_length) // stride_step) + 1
    #nrows = math.ceil(((a.size - stride_length) / stride_step) + 1)
    n = a.strides[0]
    return np.lib.stride_tricks.as_strided(a,
                                           shape=(nrows, stride_length),
                                           strides=(stride_step*n, n))

@cal_time
def framing(sig, fs=16000, win_len=0.025, win_hop=0.01):
    """
    transform a signal into a series of overlapping frames (=Frame blocking).
    Args:
        sig     (array) : a mono audio signal (Nx1) from which to compute features.
        fs        (int) : the sampling frequency of the signal we are working with.
                          Default is 16000.
        win_len (float) : window length in sec.
                          Default is 0.025.
        win_hop (float) : step between successive windows in sec.
                          Default is 0.01.
    Returns:
        array of frames.
        frame length.
    Notes:
    ------
        Uses the stride trick to accelerate the processing.
    """
    # run checks and assertions
    assert win_len > win_hop

    # compute frame length and frame step (convert from seconds to samples)
    frame_length = win_len * fs
    frame_step = win_hop * fs
    signal_length = len(sig)
    frames_overlap = frame_length - frame_step

    # make sure to use integers as indices
    frames = stride_trick(sig, int(frame_length), int(frame_step))
    if len(frames[-1]) < frame_length:
        frames[-1] = np.append(frames[-1], np.array([0]*(frame_length - len(frames[0]))))

    return frames, frame_length

@cal_time
def windowing(frames, frame_len, win_type="hamming", beta=14):
    """
    generate and apply a window function to avoid spectral leakage.
    Args:
        frames  (array) : array including the overlapping frames.
        frame_len (int) : frame length.
        win_type  (str) : type of window to use.
                          Default is "hamming"
    Returns:
        windowed frames.
    """
    if   win_type == "hamming" : windows = np.hamming(frame_len)
    elif win_type == "hanning" : windows = np.hanning(frame_len)
    elif win_type == "bartlet" : windows = np.bartlett(frame_len)
    elif win_type == "kaiser"  : windows = np.kaiser(frame_len, beta)
    elif win_type == "blackman": windows = np.blackman(frame_len)
    windowed_frames = frames * windows
    return windowed_frames

@cal_time
def trimf(x, params):
    """
    trimf: similar to Matlab definition
    https://www.mathworks.com/help/fuzzy/trimf.html?s_tid=srchtitle
    
    """
    if len(params) != 3:
        print("trimp requires params to be a list of 3 elements")
        sys.exit(1)
    a = params[0]
    b = params[1]
    c = params[2]
    if a > b or b > c:
        print("trimp(x, [a, b, c]) requires a<=b<=c")
        sys.exit(1)
    y = np.zeros_like(x, dtype=np.float32)
    if a < b:
        index = np.logical_and(a < x, x < b)
        y[index] = (x[index] - a) / (b - a)
    if b < c:    
        index = np.logical_and(b < x, x < c)              
        y[index] = (c - x[index]) / (c - b)
    y[x == b] = 1
    return y

def delta(mat):
    assert mat.ndim == 2
    win = np.array([-1.0, 0.0, 1.0]).reshape(3, 1)

    mat = np.concatenate((mat[:, :1], mat, mat[:,-1:]), axis=-1)
    mat = np.expand_dims(mat, 2)
    mat = np.concatenate((mat[:, :-2], mat[:, 1:-1], mat[:, 2:]), axis=2)

    t, v = mat.shape[:2]
    mat = np.dot(mat.reshape(-1, 3), win).reshape(t, v)

    return mat

@cal_time
def extract_lfcc(wavform, trim_filter_bank, with_energy=False, win_len=0.02, win_hop=0.01, num_ceps=20, nfft=512, with_delta=True):
    '''    
    Extracts LFCC
    '''
    # STFT
    frames, frame_length = framing(
        sig=pre_emphasis(sig=wavform, pre_emph_coeff=0.97),
        fs=16000, win_len=win_len, win_hop=win_hop)

    # -> windowing
    windows = windowing(frames=frames, frame_len=frame_length, win_type='hamming')
    spec = np.fft.fft(windows, 512)
    #  amplitude
    sp_amp = np.power(np.abs(spec[:, :nfft//2+1]), 2)
    #filter bank
    fb_feature = np.log10(np.matmul(sp_amp, trim_filter_bank)+np.finfo(np.double).eps)
    #dct
    lfccs = dct(fb_feature, type=2, axis=1, norm='ortho')[:, :num_ceps]

    #if ues energy
    if with_energy:
        power_spec = sp_amp / nfft
        energy = np.log10(power_spec.sum(axis=1)+ np.finfo(np.double).eps)
        lfccs[:, 0] = energy
    
    #dleta
    if with_delta:
        delta1 = delta(lfccs.T)
        delta2 = delta(delta1)
        feat = np.concatenate((lfccs.T, delta1, delta2), axis=0)
        feat = feat.T
        return feat
        
    else:
        return lfccs

@cal_time
def linear_fbank(nfft=512, sample_rate=16000, nfilts=20):
    # build the triangle filter bank
    f = (sample_rate / 2) * np.linspace(0, 1, nfft//2+1)
    filter_bands = np.linspace(min(f), max(f), nfilts+2)
    
    filter_bank = np.zeros([nfft//2+1, nfilts])
    for idx in range(nfilts):
        filter_bank[:, idx] = trimf(f, [filter_bands[idx], filter_bands[idx+1], filter_bands[idx+2]])
    
    return filter_bank


def extract_feat(l_utt):
    '''	
    Extracts spectrograms
    '''
    for line in l_utt:
        utt, _ = sf.read(line)
        ####normalize
        #utt = 1.0*utt/np.max(np.abs(utt))
        lfcc_fb = linear_fbank()
        spec = extract_lfcc(utt, lfcc_fb)
        spec = spec.astype(np.float32)		
        print(spec.shape)
        #dir_base, fn = os.path.split(line)
        #dir_base, faketype = os.path.split(dir_base)
        #fn, _ = os.path.splitext(fn) 
        #if not os.path.exists(os.path.join(_target_dir_base,faketype)):
        #    os.makedirs(os.path.join(_target_dir_base,faketype))
        #if not os.path.exists(os.path.join(_target_dir_base, faketype, _dir_name)):
        #    os.makedirs(os.path.join(_target_dir_base, faketype, _dir_name))
        #np.save(os.path.join(_target_dir_base,faketype, _dir_name)+fn, spec)
        dir_base, fn = os.path.split(line)
        ##!!!!
        
        dir_base, vocoder_type = os.path.split(dir_base)

        fn, _ = os.path.splitext(fn)
        # print(vocoder_type)
        if not os.path.exists(_dir_dataset + _dir_name):
            os.makedirs(_dir_dataset + _dir_name)
        np.save(_dir_dataset +_dir_name + fn, spec)

    return

_dir_dataset = './FAD/clean/train'		# directory of Dataset
_dir_name = '/lfcc/' 
_nb_proc = 24

if __name__ == '__main__':

    l_utt = []
    for r, ds, fs in os.walk(_dir_dataset):
        for f in fs:
            if os.path.splitext(f)[1] != '.wav': continue
            l_utt.append('/'.join([r, f.replace('\\', '/')]))
	
    nb_utt_per_proc = int(len(l_utt) / _nb_proc)
    l_proc = []                                                                                                                                                                                                                                                                                                                                                                                                                            
    for i in range(_nb_proc):
        if i == _nb_proc - 1:
            l_utt_cur = l_utt[i * nb_utt_per_proc :]
        else:
            l_utt_cur = l_utt[i * nb_utt_per_proc : (i+1) * nb_utt_per_proc]

        l_proc.append(Process(target = extract_feat, args = (l_utt_cur,)))
        print('%d'%i)

    for i in range(_nb_proc):
        l_proc[i].start()
        print('start %d'%i)
    for i in range(_nb_proc):
        l_proc[i].join()

    print("finished!")

