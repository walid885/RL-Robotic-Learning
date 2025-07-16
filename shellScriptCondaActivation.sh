#!/bin/bash
#inroder to make it run correctly , use this format . ./s (tab)
# Check if conda is initialized in this shell
# If not, initialize it (you might only need this if you don't have conda init in your .bashrc/.zshrc)
# If your conda is already initialized and working, you can comment out the next block.
if ! type "conda" &> /dev/null; then
    echo "Conda not found in PATH. Initializing..."
    # Replace miniconda3 with anaconda3 if that's what you're using
    source /home/walid/miniconda3/etc/profile.d/conda.sh
fi

echo "Attempting to activate FreKon environment..."
conda activate /home/walid/miniconda3/envs/FreKon

if [ $? -eq 0 ]; then
    echo "FreKon environment activated successfully."
else
    echo "Failed to activate FreKon environment."
    echo "Please ensure /home/walid/miniconda3/envs/FreKon exists and conda is properly initialized."
fi