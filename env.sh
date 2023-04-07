# Add any `module load` or `export` commands that your code needs to
# compile and run to this file.
module load languages/intel/2020-u4
module load languages/anaconda2/5.0.1
export OMP_PROC_BIND=true
export OMP_NUM_THREADS=28
export OMP_PLACES=cores
