#!/bin/bash
# Activate conda environment
# IMPORTANT: Replace with the correct path to your conda.sh if this doesn't work
# Common paths: ~/miniconda3/etc/profile.d/conda.sh or ~/anaconda3/etc/profile.d/conda.sh
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

conda activate cmp4501

# Check if activation was successful
if [ "$CONDA_DEFAULT_ENV" != "cmp4501" ]; then
    echo "Failed to activate conda environment cmp4501. Please check your conda installation and environment name."
    exit 1
fi

echo "Conda environment cmp4501 activated."

# Install matplotlib if not already installed
python -c "import matplotlib" 2>/dev/null || pip install matplotlib

python -m scripts.plot_search
python -m scripts.plot_qlearning
python -m scripts.plot_nb_alpha
python -m scripts.plot_tree_depth
python -m scripts.plot_perceptron_loss
echo "Plots saved in ./figures" 