#!/usr/bin/env python3

import os

"""
Use this commond line to run the file will keep the result in the terminal
python -u run.py > log.log 2>&1 &
log.log will keep the log in the terminal
"""

cflags = ['CFLAGS="-std=c99 -Wall -O0"', 'CFLAGS="-std=c99 -Wall -O1"', 'CFLAGS="-std=c99 -Wall -O2"', 'CFLAGS="-std=c99 -Wall -O3"', 'CFLAGS="-std=c99 -Wall -Ofast"', 'CFLAGS="-std=c99 -Wall -Ofast -mtune=native"']
# cflags = ['CFLAGS="-std=c99 -Wall -Ofast -mtune=native"']


for flag in cflags:
    if os.path.exists('d2q9-bgk'):
        os.remove('d2q9-bgk')
    print(flag)
    os.system('make '+flag)
    os.system('./d2q9-bgk input_128x128.params obstacles_128x128.dat')
    # os.remove('d2q9-bgk.out')
    # os.system('sbatch job_submit_d2q9-bgk')
    # while True:
    #     if os.path.exists('d2q9-bgk.out'):
    #         time.sleep(10)
    #         os.system('cat d2q9-bgk.out')
    #         break



print("======ALL DONE======")

