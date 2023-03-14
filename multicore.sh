for i in {1..28};
do
    echo "$i\n\n" >> multiCoreResult.txt
    export OMP_PLACES=cores
    export OMP_PROC_BIND=true
    export OMP_NUM_THREADS=$i
    sbatch -W job_submit_d2q9-bgk
    cat d2q9-bgk.out >> multiCoreResult.txt
done;