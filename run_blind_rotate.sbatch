#!/bin/bash
#SBATCH --job-name=br_n636
#SBATCH --partition=gr20100b
#SBATCH --time=3:00:00
#SBATCH --rsc p=1:t=112:c=112:m=512000
#SBATCH --output=br_n636_%j.out
#SBATCH --error=br_n636_%j.err

# Record start time
START_TIME=$(date +%s)

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Working directory: $(pwd)"

# Set the working directory
cd $SLURM_SUBMIT_DIR

# Run Blind Rotate simulation with apptainer
echo "Running Blind Rotate (n=636, stateful)..."
apptainer exec --bind $PWD:/work ~/ExactNoiseEstimation/ExactNoiseEstimation.sif \
    bash -lc 'export PATH=/usr/local/julia/bin:$PATH && cd /work && julia --project=alea -e "using Pkg; Pkg.instantiate(); Pkg.build(\"PyCall\")" && julia --project=alea blind_rotate_alea.jl --loops=636 --loop-model=stateful'

# Check if execution was successful
if [ $? -eq 0 ]; then
    echo "Blind Rotate simulation completed successfully!"
else
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    MINUTES=$(((ELAPSED % 3600) / 60))
    SECONDS=$((ELAPSED % 60))
    echo "Blind Rotate simulation failed!"
    printf "Runtime until failure: %02d:%02d:%02d (HH:MM:SS)\n" $HOURS $MINUTES $SECONDS
    echo "Runtime until failure: ${ELAPSED} seconds"
    exit 1
fi

echo "Job completed on $(date)"
