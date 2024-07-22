from typing import List

import pandas as pd
import numpy as np


from scan_table import ParsedScan

def get_df_row(parsed_scan: ParsedScan) -> List:
        return [f"{parsed_scan.hospital}_{parsed_scan.patient_num}_{parsed_scan.core}", parsed_scan.hospital, \
                parsed_scan.patient_num, parsed_scan.core, parsed_scan.label, \
                parsed_scan.psa, np.array([parsed_scan.midband_fit]), np.array([parsed_scan.spectral_slope]), \
                np.array([parsed_scan.spectral_intercept]), np.array([parsed_scan.nps]), np.array([parsed_scan.ps]), np.array([parsed_scan.ps_ref]), \
                np.array([parsed_scan.frequency_axis]), parsed_scan.pct_cancer, parsed_scan.prim_grade, parsed_scan.sec_grade, \
                parsed_scan.family_history, parsed_scan.gain, parsed_scan.depth]

def scan_table_to_df(table: List[List[ParsedScan]]) -> pd.DataFrame:
    
    cols = ["Name", "Hospital", "Patient Number", "Core", "Label", "PSA", "MBF", "SS", "SI", "NPS", "PS", "rPS", "f", 
        "PctCancer", "PrimaryGrade", "SecondaryGrade", "FamilyHistory", "Gain", "Depth"]

    df_table = []
    for file_num in range(len(table)):
        df_table.append(get_df_row(table[file_num]))

    return pd.DataFrame(df_table, columns = cols)