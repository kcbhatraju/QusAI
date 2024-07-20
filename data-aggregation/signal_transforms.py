from typing import Tuple, List, Any
from PIL import Image, ImageDraw
from pathlib import Path

import numpy as np
from numpy.matlib import repmat
from scipy.signal import hilbert

class TGCFit:
    def __init__(self, params, max_vals):
        self.fit_params: np.ndarray = params # fit between gains and max_rf_arr
        self.max_vals: np.ndarray = max_vals # [max(gains), max(max_rf_arr)]

class SpectralAnalysis:
    def __init__(self):
        self.nps: np.ndarray = None
        self.ps: np.ndarray = None
        self.ps_ref: np.ndarray = None
        self.frequency_axis: np.ndarray = None
        self.midband_fit: float = None
        self.spectral_slope: float = None
        self.spectral_intercept: float = None
        self.img_path: Path = None
        self.ref_path: Path = None
        self.location: Any = None

class ROIWindow:
    def __init__(self):
        self.left_pix: int = None
        self.right_pix: int = None
        self.top_pix: int = None
        self.bottom_pix: int = None
        self.bottom_depth_mm: float = None # relative to entire img
        self.top_depth_mm: float = None # relative to entire img
        self.width_mm: float = None

def convert_iq_rf(q_data: np.ndarray, i_data: np.ndarray, sampling_frequency: int, quad_2x: bool) -> np.ndarray:
    # check w Ahmed (reconRF.m and read_EXACTInfo.m). Centeral freq looks a little high (12-21 MHz depending on setting)
    # ignoring 8x int_fac (idk what int is). Ok? Saying RF sampling rate is 2x IQ sampling rate if quad_2x (can get up to 30MHz)
    # go over matlab code. uses two different hard-coded frequencies for this --> think wrong
    # ALSO: followed interp upsampling technique to match matlab instead of fourier-based scipy methods. Make a difference?

    fs_rf = sampling_frequency
    upsample_rate = 2 if quad_2x else 1

    n_samples = q_data.shape[0]
    n_lines = q_data.shape[1]

    i_data_int = np.zeros((n_samples*upsample_rate, n_lines))
    q_data_int = np.zeros((n_samples*upsample_rate, n_lines))
    rf_data = np.zeros((n_samples*upsample_rate, n_lines))
    t = np.arange(0,(n_samples*upsample_rate)/fs_rf, 1/fs_rf)
    if t[-1] == (n_samples*upsample_rate)/fs_rf:
        t = t[:-1]

    for i in range(n_lines):
        i_data_int[:,i] = np.interp(np.arange(0, i_data.shape[0], 1/upsample_rate), np.arange(0, i_data.shape[0]), i_data[:,i])
        q_data_int[:,i] = np.interp(np.arange(0, q_data.shape[0], 1/upsample_rate), np.arange(0, q_data.shape[0]), q_data[:,i])
        rf_data[:,i] = np.real(np.sqrt(i_data_int[:,i]**2 + q_data_int[:,i]**2) * np.sin(2*np.pi*fs_rf*np.transpose(t) + np.arctan2(q_data_int[:,i], i_data_int[:,i])))
    
    return rf_data

def max_hilbert(sub_rf_data: np.ndarray) -> int:
    hilbert_env = np.zeros_like(sub_rf_data)
    for i in range(hilbert_env.shape[1]):
        hilbert_env[:,i] = abs(hilbert(sub_rf_data[:,i]))
    return np.amax(hilbert_env)

def remove_tgc_gain(rf_data: np.ndarray, tgc_fit_params: np.ndarray, max_tgc_xy: np.ndarray, gain: float, tgc: List[float]) -> np.ndarray:
    gain = gain + np.interp(np.arange(1, rf_data.shape[0]+1), np.linspace(1, rf_data.shape[0], len(tgc)), tgc, )
    z = max_tgc_xy[1] * 10**tgc_fit_params[1] * 10**(tgc_fit_params[0]*gain/max_tgc_xy[0])
    z = np.reshape(z, (len(z), 1))
    out_rf_data = rf_data / repmat(z, 1, rf_data.shape[1])

    return out_rf_data

def compute_roi_windows(x_spline: np.ndarray, y_spline: np.ndarray, ax_pix_len_im: int, lat_pix_len_im: int, 
                        ax_mm_len_window: float, lat_mm_len_window: float, 
                        ax_res: float, lat_res, ax_overlap: float, lat_overlap: float, thresh=0.95) -> List[ROIWindow]:
    if len(x_spline) != len(y_spline):
        raise ValueError("Spline has unequal amount of x and y coordinates")
    
    # Some axial/lateral dims
    ax_pix_len_window = round(ax_mm_len_window / ax_res)
    lat_pix_len_window = round(lat_mm_len_window / lat_res)
    axial_axis = list(range(ax_pix_len_im))
    lateral_axis = list(range(lat_pix_len_im))

    # Overlap fraction determines the incremental distance between ROIs
    ax_pix_increment = ax_pix_len_window * (1 - ax_overlap)
    lat_pix_increment = lat_pix_len_window * (1 - lat_overlap)

    # Determine ROIS - Find Region to Iterate Over
    ax_start_pix = max(min(y_spline), axial_axis[0])
    ax_end_pix = min(max(y_spline), axial_axis[-1] - ax_pix_len_window)
    lat_start_pix = max(min(x_spline), lateral_axis[0])
    lat_end_pix = min(max(x_spline), lateral_axis[-1] - lat_pix_len_window)

    roi_windows = []

    # Determine all points inside the user-defined polygon that defines analysis region
    # The 'mask' matrix - "1" inside region and "0" outside region
    spline = [(x_spline[i], y_spline[i]) for i in range(len(x_spline))]
    img = Image.new("L", (lat_pix_len_im, ax_pix_len_im), 0)
    ImageDraw.Draw(img).polygon(spline, outline=1, fill=1)
    mask = np.array(img)

    for ax_pix in np.arange(ax_start_pix, ax_end_pix, ax_pix_increment):
        for lat_pix in np.arange(lat_start_pix, lat_end_pix, lat_pix_increment):
            # Convert axial and lateral positions in image indices
            axial_abs_axis = abs(axial_axis - ax_pix)
            axial_idx = np.where(axial_abs_axis == min(axial_abs_axis))[0][0]
            lateral_abs_axis = abs(lateral_axis - lat_pix)
            lateral_idx = np.where(lateral_abs_axis == min(lateral_abs_axis))[0][0]

            # Determine if ROI is Inside Analysis Region
            mask_vals = mask[
                axial_idx : (axial_idx + ax_pix_len_window),
                lateral_idx : (lateral_idx + lat_pix_len_window),
            ]

            # Define Percentage Threshold
            num_elements_in_mask = mask_vals.size
            num_elements_in_roi = len(np.where(mask_vals == 1)[0])
            window_roi_pct = num_elements_in_roi / num_elements_in_mask

            if window_roi_pct > thresh:
                roi_window = ROIWindow()
                roi_window.left_pix = int(lateral_axis[lateral_idx])
                roi_window.right_pix = int(lateral_axis[lateral_idx + lat_pix_len_window - 1])
                roi_window.top_pix = int(axial_axis[axial_idx])
                roi_window.bottom_pix = int(axial_axis[axial_idx + ax_pix_len_window - 1])
                roi_windows.append(roi_window)

    return roi_windows

def compute_power_spec(rf_data: np.ndarray, start_frequency: int, end_frequency: int, 
                       sampling_frequency: int, n_freq_points=4096) -> Tuple[np.ndarray, np.ndarray]:
    # Create Hanning Window Function
    unrm_wind = np.hanning(rf_data.shape[0])
    window_func_computations = unrm_wind * np.sqrt(len(unrm_wind) / sum(np.square(unrm_wind)))
    window_func = repmat(
        window_func_computations.reshape((rf_data.shape[0], 1)), 1, rf_data.shape[1]
    )

    frequency_range = np.linspace(0, sampling_frequency, n_freq_points)
    f_low = round(start_frequency * (n_freq_points / sampling_frequency))
    f_high = round(end_frequency * (n_freq_points / sampling_frequency))
    frequency_chop = frequency_range[f_low:f_high]

    fft = np.square(
        abs(np.fft.fft(np.transpose(np.multiply(rf_data, window_func)), n_freq_points) * rf_data.size)
    )
    full_ps = 20 * np.log10(np.mean(fft, axis=0))

    ps = full_ps[f_low:f_high]

    return frequency_chop, ps

def get_spectral_params(nps: np.ndarray, frequency_axis: np.ndarray, low_band_freq: int, up_band_freq: int) -> Tuple[float]:
    low_band_freq_dist = 999999999; up_band_freq_dist = 999999999; low_band_freq_idx = 0; up_band_freq_idx = 0

    for i in range(len(frequency_axis)):
        cur_low_band_freq_dist = abs(low_band_freq - frequency_axis[i])
        cur_up_band_freq_dist = abs(up_band_freq - frequency_axis[i])

        if cur_low_band_freq_dist < low_band_freq_dist:
            low_band_freq_dist = cur_low_band_freq_dist
            low_band_freq_idx = i

        if cur_up_band_freq_dist < up_band_freq_dist:
            up_band_freq_dist = cur_up_band_freq_dist
            up_band_freq_idx = i

    fBand = frequency_axis[low_band_freq_idx:up_band_freq_idx]
    p = np.polyfit(fBand, nps[low_band_freq_idx:up_band_freq_idx], 1)

    midband_fit = p[0] * fBand[round(fBand.shape[0] / 2)] + p[1]
    spectral_slope = p[0]
    spectral_intercept = p[1]

    return midband_fit, spectral_slope, spectral_intercept

def window_spectral_analysis(rf_data: np.ndarray, rf_data_ref: np.ndarray, min_frequency: int, max_frequency: int,
                         low_band_freq: int, up_band_freq: int, sampling_frequency: int) -> SpectralAnalysis:
    out = SpectralAnalysis()

    if max_frequency < up_band_freq:
        max_frequency = up_band_freq
    if min_frequency > low_band_freq:
        min_frequency = low_band_freq

    frequency_axis, ps = compute_power_spec(rf_data, min_frequency, max_frequency, sampling_frequency)
    frequency_axis, ps_ref = compute_power_spec(rf_data_ref, min_frequency, max_frequency, sampling_frequency)

    out.frequency_axis = frequency_axis
    out.ps = ps
    out.ps_ref = ps_ref
    out.nps = ps - ps_ref

    out.midband_fit, out.spectral_slope, out.spectral_intercept = get_spectral_params(out.nps, frequency_axis, 
                                                                                          low_band_freq, up_band_freq)
    return out