#!/bin/bash
#SBATCH --job-name=build_sif
#SBATCH --partition=eb
#SBATCH --time=01:00:00
#SBATCH --rsc p=1:t=32:c=32:m=64G
#SBATCH --output=sif_build_%j.out
#SBATCH --error=sif_build_%j.err

# Print job information
echo "Job started on $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Load necessary modules (adjust based on your cluster)
# Uncomment and adjust the following line if needed:
# module load singularity
# or
# module load apptainer

# Set the working directory
cd $SLURM_SUBMIT_DIR

# Build the Singularity/Apptainer container
echo "Building ExactNoiseEstimation container..."

# Check if singularity or apptainer is available
if command -v singularity &> /dev/null; then
    BUILD_CMD="singularity build"
    echo "Using Singularity"
elif command -v apptainer &> /dev/null; then
    BUILD_CMD="apptainer build"
    echo "Using Apptainer"
else
    echo "Error: Neither Singularity nor Apptainer found!"
    exit 1
fi

# Build the container with fakeroot if available, otherwise try without
# The --fakeroot option allows building without root privileges
if $BUILD_CMD --help | grep -q fakeroot; then
    echo "Building with fakeroot option..."
    $BUILD_CMD --force --fakeroot ExactNoiseEstimation.sif apptainer/ExactNoiseEstimation.def
else
    echo "Building without fakeroot option..."
    $BUILD_CMD --force ExactNoiseEstimation.sif apptainer/ExactNoiseEstimation.def
fi

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Container built successfully!"
    echo "Container file: ExactNoiseEstimation.sif"
    ls -lh ExactNoiseEstimation.sif
else
    echo "Container build failed!"
    exit 1
fi

echo "Job completed on $(date)"
