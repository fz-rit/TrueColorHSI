# TrueColorHSI
## Overview
Traditional methods for visualizing (previewing) hyperspectral images often use only a few selected spectral bands, which can lead to incomplete or distorted images that don't reflect how we truly see the world. These approaches also overlook how our eyes naturally perceive color.

**TrueColorHSI** solves this by using colorimetric science, standard illuminants, and standard observers to integrate the entire visible spectrum. This results in vivid, accurate images that better represent hyperspectral data and are easier for users to understand, providing a more intuitive and natural way to explore the information.


### Installation:
You can install `TrueColorHSI` via `pip`:
```bash
pip install TrueColorHSI
```

### Usage:
```python
from truecolorhsi.visualization import vanilla_visualization, colorimetric_visualization
hsi_header_file = "path/to/the/header/file"
vanilla_display_images = vanilla_visualization(header_file)
colorimetric_display_images = colorimetric_visualization(header_file, visualize=True, saveimages=True)
```

### Notes:
- This is the first official release, featuring the foundational tools for accurate hyperspectral image visualization.
- The package provides methods that help translate complex hyperspectral data into intuitive, true-to-life images that are easier to interpret and analyze.

## Example results
![Visualization from RGB bands](examples/images/Symeon/Visualization_from_rgb_bands.jpg)
*Figure 1. Visualization from appximated RGB bands (traditional method).*

![Visualization from colorimetric conversion](examples/images/Symeon/Visualization_from_colorimetric_conversion.jpg)
*Figure 2. Visualization from colorimetric conversion (our method).*

![True color visualization from different illuminants](examples/images/Symeon/Vis_from_different_illuminants.png)
*Figure 3. True color visualization using different standard illuminants (D50, D65, D75). Adjusting the chosen illuminant allows for tuning the color temperature.*

![illuminant_spd_and_CIE_xyz](examples/images/illuminant_spd_and_CIE_xyz.png)  
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
