from pathlib import Path
from typing import Tuple, List, Any

import numpy as np
import mat73
from scipy.io import loadmat
from scipy.optimize import curve_fit
from skimage.measure import find_contours

from imagistix import ImgInfo, read_exact_info, parse_iq, load_iq_img, load_metadata
from signal_transforms import TGCFit, SpectralAnalysis, convert_iq_rf, max_hilbert, remove_tgc_gain, compute_roi_windows, \
    window_spectral_analysis

CORES = ["LMM","LAM","LBM","LML","LAL","LBL","RBL","RBM","RML","RMM","RAM","RAL"]
FACILITIES = ["CRCEO","JH","PCC","PMCC","UVA"]

NEEDLE_START_DEPTH, NEEDLE_END_DEPTH = (0, 4e-2)
NEEDLE_AREA_X_LEN = 46.08e-3
NEEDLE_AREA_Y_START = 2e-3

SELECTED_FOCAL_SCAN_IDX = 1


class ParsedScan():
    def __init__(self, spectral_analyses: List[SpectralAnalysis]):
        self.nps = [analysis.nps for analysis in spectral_analyses]
        self.ps = [analysis.ps for analysis in spectral_analyses]
        self.ps_ref = [analysis.ps_ref for analysis in spectral_analyses]
        self.frequency_axis = [analysis.frequency_axis for analysis in spectral_analyses]
        self.midband_fit = [analysis.midband_fit for analysis in spectral_analyses]
        self.spectral_slope = [analysis.spectral_slope for analysis in spectral_analyses]
        self.spectral_intercept = [analysis.spectral_intercept for analysis in spectral_analyses]
        self.img_path = spectral_analyses[0].img_path
        self.ref_path = spectral_analyses[0].ref_path
        self.location = [analysis.location for analysis in spectral_analyses]

        self.gain: float = None
        self.focal_zone: int = None
        self.sampling_frequency: int = None
        self.patient_num: int = None
        self.depth: int = None
        self.label: str = None
        self.psa: float = None
        self.pct_cancer: str = None
        self.name: str = None
        self.hospital: str = None
        self.core: str = None
        self.age: int = None
        self.prim_grade: str = None
        self.sec_grade: str = None
        self.family_history: bool = None

class ImgData:
    def __init__(self):
        self.info: ImgInfo = None
        self.i_data: np.ndarray = None
        self.q_data: np.ndarray = None
        self.rf: np.ndarray = None
        self.sub_rf: np.ndarray = None
        self.max_rf: int = None

class PhantomMetaInfo:
    def __init__(self, name, gain, depth, preset):
        self.name: str = name[0]
        self.gain: int = gain[0][0] # dB
        self.depth: int = depth[0][0] # mm
        self.preset: str = preset[0]

class AnalysisParams:
    def __init__(self):
        self.axial_win_size_mm: float = None # in mm
        self.lateral_win_size_mm: float = None # in mm
        self.axial_overlap: float = None # % [0, 1]
        self.lateral_overlap: float = None # % [0, 1]
        self.min_frequency: int = None # Hz
        self.max_frequency: int = None # Hz
        self.depth30: TGCFit = None
        self.depth40: TGCFit = None
        self.depth50: TGCFit = None
        self.depth60: TGCFit = None
        self.depth70: TGCFit = None

def get_data(file_path: Path, phantoms: List[PhantomMetaInfo], frame: int) \
      -> Tuple[ImgInfo, ImgInfo, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    file_core: str = None
    file_facility: str = None

    for core in CORES:
        if core in file_path.name:
            file_core = core
            break
    if file_core is None:
        raise NameError("Core not found for inputted file")
    
    for facility in FACILITIES:
        if facility in file_path.name:
            file_facility = facility
            break
    if file_facility is None:
        raise NameError("Facility not found for inputted file")

    patient_num = file_path.name[len(file_facility)+2:len(file_facility)+5]
    label = file_path.parent.name
    data_dir = file_path.parents[2]

    img_info = read_exact_info(data_dir / Path("VXMLs") / Path(label) / Path(file_path.name[:-4]+".iq.xml"))

    if img_info.rx_gain < 35:
        img_info.match_gain = 30
    elif img_info.rx_gain >= 35 and img_info.rx_gain < 45:
        img_info.match_gain = 40
    elif img_info.rx_gain >= 45 and img_info.rx_gain < 55:
        img_info.match_gain = 50
    elif img_info.rx_gain >= 55 and img_info.rx_gain < 65:
        img_info.match_gain = 60
    elif img_info.rx_gain >= 65:
        img_info.match_gain = 70
    
    ref_phantom: PhantomMetaInfo = None
    for phantom in phantoms:
        if phantom.depth == round(img_info.depth) and phantom.gain == img_info.match_gain:
            ref_phantom = phantom
            break
    if ref_phantom is None:
        raise FileNotFoundError("Matching phantom not found")
    
    img_info_ref = read_exact_info(data_dir / Path("Phantom") / Path(ref_phantom.name[:-5]+".iq.xml"))
    i_data_ref, q_data_ref = parse_iq(img_info_ref, frame)
    img_info_ref.match_gain = img_info.match_gain

    q_data, i_data = load_iq_img(file_path)

    img_info.core = core
    img_info.id = facility
    img_info.number = patient_num
    img_info.label = label

    return img_info, img_info_ref, i_data, q_data, i_data_ref, q_data_ref

def get_needle_overlay(img: np.ndarray, depths: List[float], ymax: float, overlay_thresh: float, x0=0.0555, azimuth_ang=0.6109) -> np.ndarray:
    x = np.linspace(0, NEEDLE_AREA_X_LEN, img.shape[1])
    y = np.linspace(NEEDLE_AREA_Y_START, ymax, img.shape[0])

    x_grid, y_grid = np.meshgrid(x, y)

    e = (x0-x_grid)*np.tan(azimuth_ang) - y_grid
    d = np.sqrt((x0-x_grid)**2 + y_grid**2)

    out = (np.abs(e) < overlay_thresh) & (d > depths[0]) & (d < depths[1])
    return out

def get_window_ps(img_info: ImgInfo, img_info_ref: ImgInfo, i_data: np.ndarray, q_data: np.ndarray, 
                  i_data_ref: np.ndarray, q_data_ref: np.ndarray, analysis_params: AnalysisParams) -> List[ParsedScan]:
    
    if img_info.match_gain == 30:
        a = analysis_params.depth30.fit_params
        b = analysis_params.depth30.max_vals
    elif img_info.match_gain == 40:
        a = analysis_params.depth40.fit_params
        b = analysis_params.depth40.max_vals
    elif img_info.match_gain == 50:
        a = analysis_params.depth50.fit_params
        b = analysis_params.depth50.max_vals
    elif img_info.match_gain == 60:
        a = analysis_params.depth60.fit_params
        b = analysis_params.depth60.max_vals
    elif img_info.match_gain == 70:
        a = analysis_params.depth70.fit_params
        b = analysis_params.depth70.max_vals
    else:
        raise TypeError("Depth not found")
    if img_info.match_gain != 30:
        print("WARNING: Not in PZ Preset! Frequencies used in computation may be incorrect")

    scans_table = [None] * img_info.num_focal_zones
    for focal_zone in range(img_info.num_focal_zones):
        focal_zone_slice = np.arange(focal_zone, q_data.shape[1], img_info.num_focal_zones)
        focal_zone_slice_ref = np.arange(focal_zone, q_data_ref.shape[1], img_info_ref.num_focal_zones)
        rf_data = convert_iq_rf(q_data[:, focal_zone_slice], i_data[:, focal_zone_slice], img_info.sampling_frequency, img_info.quad_2x)
        rf_data_ref = convert_iq_rf(q_data_ref[:, focal_zone_slice_ref], i_data_ref[:, focal_zone_slice_ref], img_info_ref.sampling_frequency, img_info_ref.quad_2x)

        # from scipy.signal import hilbert
        # import matplotlib.pyplot as plt
        # bmode = np.zeros_like(rf_data)
        # for j in range(bmode.shape[1]):
        #     bmode[:,j] = 20*np.log10(abs(hilbert(rf_data[:,j])))
        
        # plt.imshow(bmode, cmap="Greys")
        # plt.show()

        # bmode = np.zeros_like(rf_data_ref)
        # for j in range(bmode.shape[1]):
        #     bmode[:,j] = 20*np.log10(abs(hilbert(rf_data_ref[:,j])))
        
        # plt.imshow(bmode, cmap="Greys")
        # plt.show()

        rf_data = remove_tgc_gain(rf_data, a, b, img_info.rx_gain, img_info.tgc)
        rf_data_ref = remove_tgc_gain(rf_data_ref, a, b, img_info_ref.rx_gain, img_info_ref.tgc)

        # bmode = np.zeros_like(rf_data)
        # for j in range(bmode.shape[1]):
        #     bmode[:,j] = 20*np.log10(abs(hilbert(rf_data[:,j])))
        
        # plt.imshow(bmode, cmap="Greys")
        # plt.show()

        # bmode = np.zeros_like(rf_data_ref)
        # for j in range(bmode.shape[1]):
        #     bmode[:,j] = 20*np.log10(abs(hilbert(rf_data_ref[:,j])))
        
        # plt.imshow(bmode, cmap="Greys")
        # plt.show()

        # Get ROI and windows within ROI - do this just the first focal zone; other images should be same size
        if not focal_zone:
            img_info.axial_res_rf = img_info.depth / rf_data.shape[0]
            img_info.lateral_res_rf = img_info.width / rf_data.shape[1]
            img_info_ref.axial_res_rf = img_info.axial_res_rf
            img_info_ref.lateral_res_rf = img_info.lateral_res_rf

            roi = get_needle_overlay(np.log10(1+abs(rf_data)), [NEEDLE_START_DEPTH, NEEDLE_END_DEPTH], img_info.depth/1000, overlay_thresh=5e-3)

            boundaries = find_contours(roi, level=0.8)
            prostate_boundary = boundaries[0]
            roi_thresh = 0.4*max(prostate_boundary[:,0])
            prostate_boundary[:,0] = np.clip(prostate_boundary[:,0], a_min=roi_thresh, a_max=np.inf)
            # prostate_boundary = prostate_boundary.astype(int)

            roi_windows = compute_roi_windows(prostate_boundary[:,1], prostate_boundary[:,0], rf_data.shape[0], rf_data.shape[1], 
                                                analysis_params.axial_win_size_mm, analysis_params.lateral_win_size_mm, 
                                                img_info.axial_res_rf, img_info.lateral_res_rf, 
                                                analysis_params.axial_overlap, analysis_params.lateral_overlap)

        analyzed_windows = []
        for window in roi_windows:
            rf_window = rf_data[window.top_pix : window.bottom_pix+1, window.left_pix : window.right_pix+1]
            rf_window_ref = rf_data_ref[window.top_pix : window.bottom_pix+1, window.left_pix : window.right_pix+1]
            window_outputs = window_spectral_analysis(rf_window, rf_window_ref, analysis_params.min_frequency, analysis_params.max_frequency,
                                                        img_info.low_band_freq, img_info.up_band_freq, img_info.sampling_frequency)
            window_outputs.location = window
            window_outputs.location.bottom_depth_mm = window.bottom_pix * img_info.axial_res_rf
            window_outputs.location.top_depth_mm = window.top_pix * img_info.axial_res_rf
            window_outputs.location.width_mm = abs(window.right_pix - window.left_pix) * img_info.lateral_res_rf
            window_outputs.img_path = img_info.file_path
            window_outputs.ref_path = img_info_ref.file_path

            analyzed_windows.append(window_outputs)
        
        parsed_scan = ParsedScan(analyzed_windows)
        parsed_scan.gain = img_info.rx_gain
        parsed_scan.depth = img_info.depth
        parsed_scan.label = img_info.label
        parsed_scan.name = img_info.file_path.name.split('_')[0]
        parsed_scan.patient_num = int(img_info.number)
        parsed_scan.focal_zone = focal_zone+1
        parsed_scan.hospital = img_info.id
        parsed_scan.core = img_info.core
        parsed_scan.sampling_frequency = img_info.sampling_frequency

        scans_table[focal_zone] = parsed_scan

    return scans_table

def read_metadata(parsed_scans: List[ParsedScan]) -> List[ParsedScan]:
    for scan in parsed_scans:
        metadata_path = scan.img_path.parents[2] / Path("Metadata") / Path(scan.label) / Path(f"{scan.name}_{scan.core}_{scan.label}.mat")
        scan.age, scan.psa, scan.family_history, scan.prim_grade, scan.sec_grade, scan.pct_cancer = load_metadata(metadata_path)
    return parsed_scans

def exact_sort(all_files: list, analysis_params: AnalysisParams, phantoms: List[PhantomMetaInfo], frame: int) -> List[List[ParsedScan]]:
    # table - (file x focal zone)
    exact_table = []
    for i, file_path in enumerate(all_files):
        if file_path.name == "phantomInfo.mat" or file_path.parents[1].name == 'Metadata':
            continue
        img_info, img_info_ref, i_data, q_data, i_data_ref, q_data_ref = get_data(file_path, phantoms, frame)
        exact_table.append(get_window_ps(img_info, img_info_ref, i_data, q_data, i_data_ref, q_data_ref, analysis_params))
        exact_table[-1] = read_metadata(exact_table[-1])
    return exact_table

def gen_scan_table(data_path: Path, frame: int) -> List[List[ParsedScan]]:
    phantom_dir = data_path / Path("Phantom")
    try:
        phantom_meta_info_matlab = loadmat(phantom_dir / Path("phantomInfo.mat"))
    except NotImplementedError:
        phantom_meta_info_matlab = mat73.loadmat(phantom_dir / Path("phantomInfo.mat"))

    phantoms = []
    for phantom_meta_info in phantom_meta_info_matlab['phantomInfo'][0]:
        phantoms.append(PhantomMetaInfo(*phantom_meta_info))

    analysis_params = AnalysisParams()
    analysis_params.axial_win_size_mm = 2 # true resolution is 0.075 mm
    analysis_params.lateral_win_size_mm = 3 # true resolution is 0.3 mm
    analysis_params.axial_overlap = 0.5
    analysis_params.lateral_overlap = 0.5
    analysis_params.min_frequency = 1000000 # best at 12 MHz (65% bandwidth for exact imaging)
    analysis_params.max_frequency = 70000000 # best at 29 MHz 

    phantom_depths = sorted(list(set(phantom.depth for phantom in phantoms)))
    phantom_gains = sorted(list(set(phantom.gain for phantom in phantoms)))

    for depth in phantom_depths:
        depth_match_phantoms = [
            phantom for phantom in phantoms if phantom.depth == depth
        ]
        depth_match_phantoms = sorted(depth_match_phantoms, key=lambda x: x.gain)

        img_data = [ImgData() for _ in range(len(depth_match_phantoms))]

        for i, phantom in enumerate(depth_match_phantoms):
            img_data[i].info = read_exact_info(phantom_dir / Path(phantom.name[:-5]+".iq.xml"))
            img_data[i].i_data, img_data[i].q_data = parse_iq(img_data[i].info, frame)
            
            focal_zone_slice = np.arange(SELECTED_FOCAL_SCAN_IDX, img_data[i].q_data.shape[1], img_data[i].info.num_focal_zones)
            img_data[i].rf = convert_iq_rf(img_data[i].q_data[:, focal_zone_slice], img_data[i].i_data[:, focal_zone_slice], img_data[i].info.sampling_frequency, img_data[i].info.quad_2x)
            img_data[i].sub_rf = img_data[i].rf[(img_data[i].rf.shape[0]//2 - 500):(img_data[i].rf.shape[0]//2 + 501), (img_data[i].rf.shape[1]//2 - 100):(img_data[i].rf.shape[1]//2 + 101)]
            img_data[i].max_rf = max_hilbert(img_data[i].sub_rf)
        
        max_rf_arr = [data.max_rf for data in img_data]

        def fun(gains: np.ndarray, m1: float, m2: float):
            return 10**m2 * 10**(m1 * gains)

        gains_norm = phantom_gains / np.max(phantom_gains)
        maxRF_norm = max_rf_arr / np.max(max_rf_arr)
        initial_guess = [1, 1]
        params, _ = curve_fit(fun, gains_norm, maxRF_norm, p0=initial_guess)

        b = [max(phantom_gains), max(max_rf_arr)]
        if depth == 30:
            analysis_params.depth30 = TGCFit(params, b)
        elif depth == 40:
            analysis_params.depth40 = TGCFit(params, b)
        elif depth == 50:
            analysis_params.depth50 = TGCFit(params, b)
        elif depth == 60:
            analysis_params.depth60 = TGCFit(params, b)
        elif depth == 70:
            analysis_params.depth70 = TGCFit(params, b)

    all_files = list(data_path.glob("**/[!._]*.mat"))

    return exact_sort(all_files, analysis_params, phantoms, frame)