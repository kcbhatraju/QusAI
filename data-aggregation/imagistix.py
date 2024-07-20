from pathlib import Path
from datetime import datetime
import xml.etree.ElementTree as ET
from typing import Tuple

import numpy as np
import mat73
from scipy.io import loadmat

FILE_HEADER = 40 # Bytes
LINE_HEADER = 4 # Bytes
INT_SIZE = 2 # Bytes
FRAME_HEADER = 56 # Bytes

# Unused and not dynamically read, but available metadata from the system
STUDY_MODE = 'RF'
SYSTEM = 'ExactImaging'
PITCH = 0
NUM_FRAMES = 1
UP_6_DB = 29000000
LOW_6_DB = 12000000
INT_FAC = 8

class ImgInfo:
    def __init__(self):
        self.bmode_focal_zone_pos1: int = None
        self.bmode_focal_zone_pos2: int = None
        self.bmode_focal_zone_pos3: int = None
        self.scan_bmode_focal_zone_pos1: int = None
        self.scan_bmode_focal_zone_pos2: int = None
        self.scan_bmode_focal_zone_pos3: int = None
        self.rx_gain: float = None
        self.tx_power: float = None
        self.prf: int = None # Pulse Rep Freq
        self.zoom_height: int = None
        self.tx_fnum: float = None
        self.tgc_fixed_gain: float = None
        self.fixed_gain: float = None
        self.line_density_usr: float = None
        self.line_density: float = None
        self.preset: str = None
        self.probe: str = None
        self.study_name: str = None
        self.mode_name: str = None
        self.acq_datetime = None
        self.data_format: str = None
        self.user_gain: float = None
        self.tgc1: float = None
        self.tgc2: float = None
        self.tgc3: float = None
        self.tgc4: float = None
        self.tgc5: float = None
        self.tgc6: float = None
        self.tgc7: float = None
        self.tgc8: float = None
        self.study_mode: str = None
        self.file_path: Path = None
        self.system: str = None
        self.samples: int = None
        self.lines: int = None
        self.depth_offset: int = None
        self.depth: int = None
        self.width: float = None
        self.rx_frequency: int = None
        self.center_frequency: int = None
        self.quad_2x: bool = None
        self.num_focal_zones: int = None
        self.frame_size: int = None
        self.depth_axis: int = None
        self.width_axis: int = None
        self.axial_res: float = None
        self.lateral_res: float = None
        self.dyn_range: int = None
        self.bmode_y_offset: int = None
        self.bmode_v_offset: int = None
        self.low_band_freq: int = None
        self.up_band_freq: int = None
        self.q_2x_frequency: int = None
        self.sampling_frequency: int = None
        self.tgc: list = None
        self.match_gain: int = None
        self.core: str = None
        self.id: str = None
        self.number: int = None
        self.label: str = None
        self.axial_res_rf: float = None
        self.lateral_res_rf: float = None

def get_image_info(root: ET.Element) -> ImgInfo:
    info = ImgInfo()
    for k in range(len(root)):
        node = root[k].get('name')
        if node == 'B-Mode/Display-Range':
            info.dyn_range = int(root[k].get('value'))                                                       
        elif node == 'B-Mode/V-Offset':
            info.bmode_v_offset = int(root[k].get('value'))
        elif node == 'B-Mode/Y-Offset':
            info.bmode_y_offset = int(root[k].get('value'))    
        elif node == 'B-Mode/Samples':
            info.samples = int(root[k].get('value'))
        elif node == 'B-Mode/Lines':
            info.lines = int(root[k].get('value'))      
        elif node == 'B-Mode/Depth-Offset':
            info.depth_offset = int(root[k].get('value'))     
        elif node == 'B-Mode/Depth':
            info.depth = int(root[k].get('value'))       
        elif node == 'B-Mode/Width':
            info.width = float(root[k].get('value'))     
        elif node == 'B-Mode/RX-Frequency':
            info.rx_frequency = int(root[k].get('value'))
        elif node == 'B-Mode/TX-Frequency':
            info.center_frequency = int(root[k].get('value'))          
        elif node == 'B-Mode/Quad-2X':
            info.quad_2x = (root[k].get('value') == 'true')
        elif node == 'B-Mode/Focal-Zones-Count':
            info.num_focal_zones = int(root[k].get('value'))
        elif node == 'B-Mode/Focal-Zones-Pos1':
            info.bmode_focal_zone_pos1 = int(root[k].get('value'))
        elif node == 'B-Mode/Focal-Zones-Pos2':
            info.bmode_focal_zone_pos2 = int(root[k].get('value'))
        elif node == 'B-Mode/Focal-Zones-Pos3':
            info.bmode_focal_zone_pos3 = int(root[k].get('value'))
        elif node == 'B-Mode/Scan/Focal-Zones-Pos1':
            info.scan_bmode_focal_zone_pos1 = int(root[k].get('value'))
        elif node == 'B-Mode/Scan/Focal-Zones-Pos2':
            info.scan_bmode_focal_zone_pos2 = int(root[k].get('value'))
        elif node == 'B-Mode/Scan/Focal-Zones-Pos3':
            info.scan_bmode_focal_zone_pos3 = int(root[k].get('value'))
        elif node == 'B-Mode/RX-Gain':
            info.rx_gain = float(root[k].get('value'))
        elif node == 'B-Mode/TX-Power':
            info.tx_power = float(root[k].get('value'))
        elif node == 'B-Mode/TX/Pulse-Rep-Frequency/hack':
            info.prf = int(root[k].get('value'))
        elif node == 'B-Mode/Ctr-Frequency':
            pass # same as tx-frequency
        elif node == 'B-Mode/Zoom-Height':
            info.zoom_height = int(root[k].get('value'))
        elif node == 'B-Mode/TX-FNum':
            info.tx_fnum = float(root[k].get('value'))
        elif node == 'B-Mode/TGC-Fixed-Gain':
            info.tgc_fixed_gain = float(root[k].get('value'))
        elif node == 'B-Mode/Fixed-Gain':
            info.fixed_gain = float(root[k].get('value'))
        elif node == 'B-Mode/Line-Density-User':
            info.line_density_usr = float(root[k].get('value'))
        elif node == 'B-Mode/Line-Density':
            info.line_density = float(root[k].get('value'))
        elif node == 'Preset':
            info.preset = root[k].get('value')   
        elif node == 'Transducer-Name':
            info.probe = root[k].get('value')
        elif node == 'Study-Name':
            info.study_name = root[k].get('value')
        elif node == 'Mode-Name':
            info.mode_name = root[k].get('value')
        elif node == 'Acquired-Date':
            date = np.array(root[k].get('value').split('-')).astype(np.uint64)
        elif node == 'Acquired-Time':
            time = np.array(root[k].get('value').split(':'))
        elif node == 'Data-Format':
            info.data_format = root[k].get('value')
        elif node == 'Rx/User-Gain':
            info.user_gain = float(root[k].get('value'))
        elif node == 'Tgc/Control[1]':
            info.tgc1 = float(root[k].get('value'))
        elif node == 'Tgc/Control[2]':
            info.tgc2 = float(root[k].get('value'))
        elif node == 'Tgc/Control[3]':
            info.tgc3 = float(root[k].get('value'))
        elif node == 'Tgc/Control[4]':
            info.tgc4 = float(root[k].get('value'))
        elif node == 'Tgc/Control[5]':
            info.tgc5 = float(root[k].get('value'))
        elif node == 'Tgc/Control[6]':
            info.tgc6 = float(root[k].get('value'))
        elif node == 'Tgc/Control[7]':
            info.tgc7 = float(root[k].get('value'))
        elif node == 'Tgc/Control[8]':
            info.tgc8 = float(root[k].get('value'))

    info.depth_axis = np.arange(info.depth_offset, info.depth + (info.depth - info.depth_offset)/(info.samples-1), (info.depth - info.depth_offset)/(info.samples-1))
    info.width_axis = np.arange(0, info.width + info.width/(info.lines-1), info.width/(info.lines-1))          
    info.axial_res = info.depth/info.samples
    info.lateral_res = info.width/info.lines
    info.frame_size = info.axial_res*info.lateral_res
    info.acq_datetime = datetime(int(date[0]), int(date[1]), int(date[2]), int(time[0]), int(time[1]), int(time[2].split('.')[0]))

    info.low_band_freq = info.center_frequency - 0.65*info.center_frequency
    info.up_band_freq = info.center_frequency + 0.65*info.center_frequency
    
    info.study_mode = STUDY_MODE
    info.system = SYSTEM

    if info.quad_2x:
        info.q_2x_frequency = info.rx_frequency*2
    else:
        info.q_2x_frequency = info.rx_frequency
    # info.sampling_frequency = info.q_2x_frequency*INT_FAC
    info.sampling_frequency = info.q_2x_frequency*INT_FAC
    info.tgc = [info.tgc1, info.tgc2, info.tgc3, info.tgc4, info.tgc5, info.tgc6, info.tgc7, info.tgc8]
    return info

def read_exact_info(file_path: Path) -> ImgInfo:
    tree = ET.parse(f"{file_path}")
    root = tree.getroot()
    mode = root[0].get('value')

    if mode != 'B-Mode':
        raise TypeError("Error: Not B-Mode scan")
    info = get_image_info(root)
    info.file_path = file_path
    return info

def parse_iq(info: ImgInfo, frame: int) \
      -> Tuple[np.ndarray, np.ndarray]:
    n_lines = info.num_focal_zones * info.lines
    i_data = np.zeros((info.samples, n_lines))
    q_data = np.zeros((info.samples, n_lines))
    header_len = FILE_HEADER + (FRAME_HEADER*frame)
    header_len += (frame-1) * (INT_SIZE*info.samples*n_lines*2 + LINE_HEADER*n_lines)
    bmode_path = info.file_path.parent / Path(info.file_path.name[:-4]+".bmode")

    f = open(bmode_path, 'rb')
    for i in range(n_lines):
        f.seek(header_len + (INT_SIZE*info.samples*2 + LINE_HEADER)*i, 0)
        f.seek(LINE_HEADER, 1)
        for j in range(info.samples):
          q_data[j,i] = int.from_bytes(f.read(INT_SIZE), 'little', signed=True)
          f.seek(INT_SIZE, 1)
        f.seek(header_len + (INT_SIZE*info.samples*2 + LINE_HEADER)*i + INT_SIZE, 0)
        f.seek(LINE_HEADER, 1)
        for j in range(info.samples):
            i_data[j,i] = int.from_bytes(f.read(INT_SIZE), 'little', signed=True)
            f.seek(INT_SIZE, 1)
    
    return i_data, q_data

def load_metadata(metadata_path: Path) -> Tuple[int, float, bool, str, str, str]:
    try:
        metadata = loadmat(metadata_path)['metadata']
        age, psa, family_history = metadata['Age'][0], metadata['PSA'][0], (metadata['FamilyHistory'][0] == 'True')
        primary_grade, secondary_grade, pct_cancer = metadata['PrimaryGrade'][0], metadata['SecondaryGrade'][0], metadata['PctCancer'][0]
    except NotImplementedError:
        metadata = mat73.loadmat(metadata_path)['metadata']
        age, psa, family_history = metadata['Age'], metadata['PSA'], (metadata['FamilyHistory'] == 'True')
        primary_grade, secondary_grade, pct_cancer = metadata['PrimaryGrade'], metadata['SecondaryGrade'], metadata['PctCancer']

    if len(age): age = int(age)
    if len(psa): psa = float(psa)
    if len(primary_grade): primary_grade = float(primary_grade)
    if len(secondary_grade): secondary_grade = float(secondary_grade)
    if len(pct_cancer): pct_cancer = float(pct_cancer)
    
    return age, psa, family_history, primary_grade, secondary_grade, pct_cancer
    
def load_iq_img(file_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    try:
        img_iq = loadmat(file_path)
        i_data = img_iq['I'][0]
        q_data = img_iq['Q'][0]
    except NotImplementedError:
        img_iq = mat73.loadmat(file_path)
        i_data = img_iq['I']
        q_data = img_iq['Q']
    return q_data, i_data