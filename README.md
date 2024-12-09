# TrueColorHSI
## Overview
TrueColorHSI is a Python toolkit that turns hyperspectral images into color visuals that reflect how we truly see the world. Unlike traditional methods that rely on a few chosen bands, it uses colorimetric science, standard illuminants, and standard observers to integrate over the whole visible spectrum. It generates vivid, accurate images that help users better understand and explore the input hyperspectral data.


## Setup Instructions

```bash
# Clone the repository, checkout the Github_Cloning_Guide.md if needed.
git clone https://github.com/fz-rit/TrueColorHSI.git

# Navigate to the project directory
cd TrueColorHSI

# Create a new conda environment with Python 3.9 (not the latest python), since it's relatively more compatible to different packages as of 11/14/2024.
conda create -n truecolorhsi_env python=3.9

# Activate the newly created environment
conda activate truecolorhsi_env

# Install the dependencies with pip
pip install spectral matplotlib scipy scikit-image pysptools
pip install colour-science
pip install huggingface_hub
huggingface-cli login # To login, create and add your token according to the guide
```

---
## Files description
| File/Folder                         | Description |
|-------------------------------------|-------------|
| `Accessories_from_colour.py`        | Tool function of getting the spectral power distribution of illuminant. |
| `colorimetric_based_visualization_of_hyperspec.py`           | The true color visualization script. |

---

## Usage
1) Clone the repo and setup the environment;
2) [Optional: if you want to run the test dataset] Clone the test dataset to a specific folder:
```bash
git clone fz-rit-hf/rit-cis-hyperspectral-Symeon # you may need to request for access. 
```
3) Adjust the input paths in lines 223 and 224 in the `colorimetric_based_visualization_of_hyperspec.py` file. Then run the script: 
```bash 
python colorimetric_based_visualization_of_hyperspec.py
```


## Citations
If you find this repository useful in your research, please consider the following citations.
### Maximum Distance
```bib
@article{amiri2024colorimetric,
  title={Colorimetric characterization of multispectral imaging systems for visualization of historical artifacts},
  author={Amiri, Morteza Maali and Messinger, David W and Hanneken, Todd R},
  journal={Journal of Cultural Heritage},
  volume={68},
  pages={136--148},
  year={2024},
  publisher={Elsevier}
}
```
