export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export MKL_DOMAIN_NUM_THREADS=1
export JULIA_NUM_THREADS=1

pytest benchmarks.py --benchmark-storage="file://data" \
    --benchmark-save="data" --benchmark-sort=name --benchmark-min-rounds=5 \
    > "log.out" 2> "log.err"
