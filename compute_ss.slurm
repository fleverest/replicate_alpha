#!/bin/bash
#SBATCH --job-name=Table2_Sample_Sizes
#SBATCH --time=5-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=floyd.everest@monash.edu
#SBATCH -e log/table2.%a.%A.err
#SBATCH -o log/table2.%a.%A.out

module load miniforge3
mamba activate alpha-env
source .venv/bin/activate

time python3 compute_ss.py
