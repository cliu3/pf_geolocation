#/bin/bash
mkdir results
python -m cProfile -s cumtime -o profile_gpu.cprof ../../pf_geolocation/run_pf_gpu.py >>out_gpu.log
mv result*.mat results/gpu.mat
python -c "import pstats;stats = pstats.Stats('profile_gpu.cprof');stats.strip_dirs().sort_stats('cumtime').print_stats(20)" > profile_out_gpu

