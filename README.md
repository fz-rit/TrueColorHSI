# TrueColorHSI
## Overview

Traditional hyperspectral visualization methods convert images to RGB by averaging bands into fixed ranges corresponding to blue, green, and red. While practical, this method oversimplifies the data and may result in a loss of important details and nuances.

**TrueColorHSI** takes a more sophisticated approach by using colorimetric science to process the entire visible spectrum, delivering vivid, perceptually accurate images. Additionally, it offers users the flexibility to adjust the illuminant (D50, D55, D65, D75), enhancing the interpretation of hyperspectral data under different lighting conditions.

### Updated Comparison Table: Traditional Method vs. **TrueColorHSI**  

| **Aspect**              | **Traditional Method**                                              | **✨ TrueColorHSI ✨**                                              |
|-------------------------|---------------------------------------------------------------------|--------------------------------------------------------------------|
| **Spectral Band Usage**  | ⚙️ Fixed RGB ranges (Blue, Green, Red)                              | 🌈 **Full visible spectrum utilization**                           |
| **Color Basis**          | ⚙️ Based on peak wavelengths (~470, ~545, ~680 nm)                 | 🌈 **Colorimetric science-based**                                  |
| **Color Accuracy**       | ⚠️ Approximate color reproduction                                   | ✅ **Highly accurate color representation**                         |
| **Visualization Quality**| ⚠️ Simplified, may lose details                                    | ✅ **Vivid and detailed output**                                    |
| **User Experience**      | ⚙️ Limited user control                                            | ✅ **Tunable illuminants (D50, D55, D65, D75)**                    |

---

Note: The input data need to cover the full visible spectrum (400-700 nm) to achieve the best results. If the data is not full-spectrum, the package will still work, but the output may not be as accurate.


### Installation:
You can install `TrueColorHSI` via `pip`:
```bash
pip install TrueColorHSI
```

### Usage:


```python
from truecolorhsi.visualization import vanilla_visualization, colorimetric_visualization
from pathlib import Path
input_path = Path("path/to/the/input/file")
vanilla_display_images = vanilla_visualization(input_path)
colorimetric_display_images = colorimetric_visualization(input_path, visualize=True, saveimages=True)
```
Supportted data format:
- [Symeon-Cultural-Heritage](https://huggingface.co/datasets/fz-rit-hf/rit-cis-hyperspectral-Symeon) ENVI HSI data, which comes with a `.hdr` file along with the data file (prefered)
- [Heidelberg Porcine HyperSPECTRAL Imaging Dataset](https://heiporspectral.org/) - a binary file ends with `.dat`
  - For more details about tests, checkout the note book [test_visualization_bio.ipynb](notebooks/test_visualization_bio.ipynb)

### Notes:
- The package provides methods that help translate complex hyperspectral data into intuitive, true-to-life images that are easier to interpret and analyze.

## Example results
<p align="center">
  <img src="examples/Symeon_VNIR_cropped/Visualization_from_rgb_bands.jpg" alt="Visualization from RGB bands" width="45%">
  <img src="examples/Symeon_VNIR_cropped/Visualization_from_colorimetric_conversion_D65.jpg" alt="Visualization from colorimetric conversion" width="45%">
</p>
<p align="center">
  <em>Figure 1. Visualization from approximated RGB bands (traditional method) on the left. Figure 2. Visualization from colorimetric conversion (our method) on the right.</em>
</p>

![True color visualization from different illuminants](examples/Vis_from_different_illuminants.png)
*Figure 3. True color visualization using different standard illuminants (D50, D65, D75) (white balance not applied). Adjusting the chosen illuminant allows for tuning the color temperature.*

![illuminant_spd_and_CIE_xyz](examples/illuminant_spd_and_CIE_xyz.png)  
*Figure 4. The spectral power distribution of the D65 illuminant and the CIE xyz curves.*



## Citation
If you find this repository useful in your research, please consider the following citation.
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
