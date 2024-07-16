from typing import List

import pandas as pd


from scan_table import ParsedScan

def get_df_row(parsed_scan: ParsedScan) -> List:
        return [parsed_scan.name, parsed_scan.focal_zone, parsed_scan.label, parsed_scan.psa, \
                parsed_scan.midband_fit, parsed_scan.spectral_slope, parsed_scan.spectral_intercept, \
                parsed_scan.nps, parsed_scan.ps, parsed_scan.ps_ref, parsed_scan.frequency_axis, 
                parsed_scan.gain, parsed_scan.depth, parsed_scan.core, parsed_scan.hospital]

def scan_table_to_df(table: List[List[ParsedScan]]) -> pd.DataFrame:
    
    cols = ["Name", "Focal Zone", "Label", "PSA", "MBF", "SS", "SI", "NPS", "PS", "rPS", "f", 
        "Gain", "Depth", "Core", "Hospital"]

    df_table = []
    for file_num in range(len(table)):
        for focal_zone_num in range(len(table[file_num])):
            df_table.append(get_df_row(table[file_num][focal_zone_num]))

    return pd.DataFrame(df_table, columns = cols)