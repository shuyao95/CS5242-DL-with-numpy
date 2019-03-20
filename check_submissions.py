import numpy as  np 
import pandas as pd 
import os
import argparse

parser = argparse.ArgumentParser('Check the usbmission of students')
parser.add_argument('--sub_dir', type=str, default='~/Downloads',
                    help='the directory of students submission')
parser.add_argument('--list_path',type=str, default='~/Downloads',
                    help='the path of students list')

args = parser.parse_args()

students = pd.read_csv(args.list_path, sep='\t', header=None).values
files = os.listdir(args.sub_dir)
files_id = [str.lower(file.split('.')[0]) for file in files]

print('-'*50)
print('Not submited students:')
for stu in students:
    found = -1
    for file_id in files_id:
        if str.lower(stu[0]) in file_id or str.lower(stu[1]) in file_id:
            found = 1
            break
    if found == -1:
        print(stu[0], stu[1])

print('-'*50)
print('Submitted files by guest students:')
for file_id in files_id:
    found = -1
    for stu in students:
        if file_id in str.lower(stu[0]) or file_id in str.lower(stu[1]):
            found = 1
    if found == -1:
        print(file_id)
