
# check python
conda deactivate
export PATH="/opt/conda/bin:$PATH"
echo "Conda environment deactivated. Current Python version is: $(which python)"
echo "Python version: $(python --version)"
pip list

