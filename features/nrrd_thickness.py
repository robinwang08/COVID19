import os
from glob import glob
import nrrd

nrrd_folder= '/home/user1/COVID_unprocessed_nrrd'

def get_nrrd_name(nrrd_path):
    directories = nrrd_path.split('\\')
    file_name = directories[len(directories)-1]
    nrrd_name = file_name.split('.nrrd')[0]
    return nrrd_name

def get_thickness(nrrd_path, decimals=-1):
    data, header = nrrd.read(nrrd_path)
    thickness = header.get('space directions')[2][2]
    round_digits = decimals > -1
    if round_digits:
        thickness = round(thickness,decimals)
    return thickness

csvfile = open('thickness.csv','w+')
csvfile.write('case, thickness\n')

cp = os.path.normpath(nrrd_folder)
scan_list = glob(cp+'/*')

for scan_directory in scan_list:
    case_name = get_nrrd_name(scan_directory)
    thickness = get_thickness(scan_directory, decimals=3)
    entry = case_name+', '+str(thickness)+'\n'
    csvfile.write(entry)

csvfile.close()