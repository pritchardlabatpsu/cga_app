# The individual source data .csv files are generated in `figures.py` and `figures.R`. This 
# script consolidates all the csv's into a single Excel file

import os
import glob
import pandas as pd

# output directory
dir_out = './manuscript/figures/'
dir_out_srcdata = os.path.join(dir_out, 'source_data')

# consolidate
file_list = glob.glob(os.path.join(dir_out_srcdata, '*.csv'))
file_dict = {}
for csv_file in file_list:
    tab_name = os.path.basename(csv_file).split('.csv')[0]
    val_replace_dict = {'fig_2c-e_gene':'fig_2c', 'fig_2c-e_paralog':'fig_2d', 'fig_2c-e_Panther':'fig_2e'}
    for k,v in val_replace_dict.items():
        tab_name = tab_name.replace(k,v)
    file_dict[csv_file] = tab_name

writer = pd.ExcelWriter(os.path.join(dir_out_srcdata, 'source_data.xlsx'), engine='xlsxwriter')
format_header = writer.book.add_format({'bold': False, 'align': 'left', 'valign': 'top'})
for csv_file, tab_name in sorted(file_dict.items(), key=lambda x: x[1]):
    # read in csv, and write to excel sheet
    print(f"Reading {csv_file}")
    df = pd.read_csv(csv_file)
    df.to_excel(writer, sheet_name=tab_name, index=False)

    # format header
    worksheet = writer.sheets[tab_name]
    for col_num, col_val in enumerate(df.columns.values):
        if col_val.startswith("Unnamed:"):
            t = worksheet.write(0, col_num, '', format_header)
        else:
            t = worksheet.write(0, col_num, col_val[0].upper()+col_val[1:], format_header)

writer.save()
