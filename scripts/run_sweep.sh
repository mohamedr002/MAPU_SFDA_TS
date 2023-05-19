
python ../main.py --is_sweep True --num_sweeps 100 --experiment_description HAR --run_description MAPU --da_method MAPU --dataset HAR --sweep_name MAPU_HAR
python ../main.py --is_sweep True --num_sweeps 100 --experiment_description HAR --run_description MAPU --da_method DebugSfda --dataset EEG --sweep_name MAPU_SSC
python ../main.py --is_sweep True --num_sweeps 100 --experiment_description HAR --run_description MAPU --da_method DebugSfda --dataset SSC --sweep_name MAPU_FD
