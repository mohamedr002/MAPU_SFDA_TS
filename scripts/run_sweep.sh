
python trainers/sweep.py --num_sweeps 100 --experiment_description MAPU_HAR --run_description MAPU --da_method MAPU --dataset HAR --sweep_name MAPU_HAR
python trainers/sweep.py  --num_sweeps 100 --experiment_description MAPU_SSC --run_description MAPU --da_method MAPU --dataset EEG --sweep_name MAPU_SSC
python trainers/sweep.py  --num_sweeps 100 --experiment_description MAPU_FD --run_description MAPU --da_method MAPU --dataset FD --sweep_name MAPU_FD
