# -*- coding: utf-8 -*-
"""
External File:
- Accessories.py

@author: Morteza, David Messenger, Fei Zhang
"""

import numpy as np
import matplotlib.pyplot as plt
import colour
import skimage.exposure as exposure
from scipy.interpolate import interp1d
from pathlib import Path
import skimage
from typing import Optional, Union, List
from truecolorhsi.accessories import get_illuminant_spd_and_xyz, get_band_index, percentile_stretching, read_HSI_data, white_balance
# 
# from accessories import get_illuminant_spd_and_xyz, get_band_index, percentile_stretching, read_HSI_data, white_balance



def skimage_clahe_for_color_image(image: np.ndarray) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to a color image using skimage.
    Convert the image to LAB color space, apply CLAHE to the L channel, and convert back to RGB.

    Parameters:
    image: the input color image

    Returns:
    equalized_image: the color image after applying CLAHE
    """
    # Convert to LAB color space
    lab_image = skimage.color.rgb2lab(image)

    # Normalize the L (luminance) channel to [0, 1]
    l_channel = lab_image[..., 0] / 100.0

    # Apply CLAHE to the normalized L channel
    l_channel_eq = exposure.equalize_adapthist(l_channel)

    # Rescale the L channel back to [0, 100]
    lab_image[..., 0] = l_channel_eq * 100.0

    # Convert back to RGB color space
    equalized_image = skimage.color.lab2rgb(lab_image)

    # Clip the values to [0, 1]
    equalized_image = np.clip(equalized_image, 0, 1)

    return equalized_image

def vanilla_hsi_to_rgb(hsi_cube: np.ndarray, 
               band_centers: np.ndarray,
               input_path: Path) -> np.ndarray:
    """
    Convert the hyperspectral image to RGB by averaging adjacent bands within certain ranges to 
    ensure robust RGB reconstruction. The ranges are chosen based on the peak wavelengths
    (~470 nm, ~545 nm, and ~680 nm), resulting in 450-495 nm, 495-570 nm, and 620-750 nm for blue,
    green, and red, respectively. These ranges are based on human perception of visible light and are 
    commonly used in display technologies.

    Parameters:
    hsi_cube: np.ndarray
        The hyperspectral image cube.
    band_centers: np.ndarray
        The center wavelengths of the bands in the hyperspectral image.

    Returns:
    np.ndarray
        The RGB image.
    """
    
    if input_path.suffix == ".dat":
        blue_range = (530, 560)
        green_range = (540, 590)
        red_range = (585, 725)
    else:
        blue_range = (450, 495)
        green_range = (500, 575)
        red_range = (620, 750)
    

    iblue_start = get_band_index(band_centers, blue_range[0])
    iblue_end = get_band_index(band_centers, blue_range[1])
    igreen_start = get_band_index(band_centers, green_range[0])
    igreen_end = get_band_index(band_centers, green_range[1])
    ired_start = get_band_index(band_centers, red_range[0])
    ired_end = get_band_index(band_centers, red_range[1])
    
    
    # Average pixel values in the specified ranges
    blue = np.mean(hsi_cube[:, :, iblue_start:iblue_end+1], axis=2)
    green = np.mean(hsi_cube[:, :, igreen_start:igreen_end+1], axis=2)
    red = np.mean(hsi_cube[:, :, ired_start:ired_end+1], axis=2)
    print("- RGB bands extracted from the hyperspectral image.")
    print(f"  Aggregated bands: \n"
          f"    blue:[{band_centers[iblue_start : iblue_end+1]} nm] \n"
          f"    green:[{band_centers[igreen_start : igreen_end+1]} nm] \n"
          f"    red:[{band_centers[ired_start : ired_end+1]} nm]")
    
    # Stack channels to form an RGB image
    viz_simple = np.stack((red, green, blue), axis=-1)
    return viz_simple

def make_compare_plots(images: tuple[np.ndarray, np.ndarray],
                       suptitle: str, 
                       subplot_titles: List[str],
                       saveimages: bool, 
                       savefolder: Path) -> None:
    """
    Make a comparison plot of the input images.

    Parameters:
    images: a tuple of two images to be compared
    suptitle: the title of the plot
    subplot_titles: the title of each subplot
    saveimages: whether to save the plot as an image
    savefolder: the folder to save the image

    Returns:
    None
    """
    len_images = len(images)
    fig, axes = plt.subplots(len_images, 1, figsize=(8, 6*len_images))

    for i, image in enumerate(images):
        axes[i].imshow(image)
        axes[i].axis('off')
        axes[i].set_title(f'{subplot_titles[i]}')

    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    if saveimages:
        outfile = savefolder / f'{suptitle}.jpg'
        print('Writing to: ', outfile)
        plt.savefig(outfile, bbox_inches = 'tight', dpi = 300)

    plt.show()

def vanilla_visualization(input_path: Union[str, Path],
                          visualize: bool = False,
                          stretch_percent: int = 2,
                          saveimages: bool = True,
                          savefolder: Optional[Path] = None,) -> tuple[np.ndarray, np.ndarray]:
    """
    Display the hyperspectral image by directly visualizing the RGB bands.

    Parameters:
    header_file: the header file of the hyperspectral image
    saveimages: whether to save the plot as an image
    savefolder: the folder to save the image

    Returns:
    display_images: a tuple of the original RGB image and the contrast-enhanced RGB image
    """
    print("=============Vanilla Visualization===============")
    hyperspec_cube, band_centers = read_HSI_data(input_path)

    viz_rgb = vanilla_hsi_to_rgb(hyperspec_cube, band_centers, input_path)
    print("- RGB image extracted from the hyperspectral image.")
    viz_norm = exposure.rescale_intensity(viz_rgb, out_range=(0, 1))
    print("- RGB image normalized to (0, 1) to fit into the sRGB range.")    

    viz_stretch = percentile_stretching(viz_norm, stretch_percent)
    print(f"- Percentile stretching applied to the RGB image. (%{stretch_percent})")

    viz_norm_wb = white_balance(viz_stretch, gamut_mapping=1)
    print("- White balance applied to the RGB image.")

    # Apply more advanced contrast stretch: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    viz_clahe_on_L = skimage_clahe_for_color_image(viz_norm_wb)
    print("- CLAHE applied to the RGB image.")

    display_images = (viz_norm, viz_stretch, viz_norm_wb, viz_clahe_on_L)
    subplot_titles = ['RGB', f'RGB ({stretch_percent}% stretch)', 'RGB (stretch+wb)', 'RGB (stretch+wb+CLAHE)']
    if visualize:
        savefolder = input_path.parent / 'outputs' if savefolder is None else savefolder
        make_compare_plots(images=display_images, 
                        suptitle='Visualization_from_rgb_bands', 
                        subplot_titles=subplot_titles,
                        saveimages=saveimages, 
                        savefolder=savefolder)

    return display_images


def colorimetric_visualization(input_path: Union[str, Path], 
                               illuminant: str = 'D65',
                               stretch_percent: int = 2,
                               visualize: bool = False,
                               saveimages: bool = True, 
                               savefolder: Optional[Path] = None, ) -> tuple[np.ndarray, np.ndarray]:
    """
    Display the hyperspectral image by converting the reflectance data to sRGB using colorimetric methods.

    Parameters:
    input_path: the header file of the hyperspectral image
    illuminant: the illuminant used for colorimetric conversion
    stretch_percent: the percentage of contrast stretch
    saveimages: whether to save the plot as an image
    savefolder: the folder to save the image

    Returns:
    display_images: a tuple of the original sRGB image and the contrast-enhanced sRGB image

    """
    print("=============Colorimetirc Visualization===============")
    hyperspec_cube, band_centers = read_HSI_data(input_path)

    #Interpolating the standard data of standard illuminant and 
    #standard observer to coincide with the wavelengths that
    #our hyperspectral image has
    nrows, ncols, nbands = hyperspec_cube.shape
    print(f'IMAGE rows, cols, bands: {(nrows, ncols, nbands)}')
    
    i_cutoff = get_band_index(band_centers, 830.0)
    hyperspec_wavelengths = band_centers[:i_cutoff]
    print(f"Bands used for colorimetric conversion: {hyperspec_wavelengths}")

    std_wavelengths, illuminant_values, xyz = get_illuminant_spd_and_xyz(illuminant=illuminant, plot_flag=False, run_example=False)

    # Create an interpolation function based on spectral power distribution of illuminant
    interp_function = interp1d(std_wavelengths, illuminant_values, kind='linear', fill_value="extrapolate")

    # Interpolate the illuminant data to match the wavelengths of the hyperspectral image
    illuminant_interp = interp_function(hyperspec_wavelengths)

    # Create three interpolation functions based on the standard observer tristimulus values.
    interp_func_0 = interp1d(std_wavelengths, xyz[:, 0], kind='linear', fill_value='extrapolate')
    interp_func_1 = interp1d(std_wavelengths, xyz[:, 1], kind='linear', fill_value='extrapolate')
    interp_func_2 = interp1d(std_wavelengths, xyz[:, 2], kind='linear', fill_value='extrapolate')

    # Get the coreesponding tristimulus values for the wavelengths of the hyperspectral image
    cie_x_interp = interp_func_0(hyperspec_wavelengths)
    cie_y_interp = interp_func_1(hyperspec_wavelengths)
    cie_z_interp = interp_func_2(hyperspec_wavelengths)
    xyz_interp = np.column_stack((cie_x_interp, cie_y_interp, cie_z_interp)) # shape 186x3
    

    # Get the reflectance data in the visible range
    visible_range_data = hyperspec_cube[:, :, :i_cutoff].reshape((-1, i_cutoff))

    # Convert Reflectance to CIEXYZ tristimulus values
    XYZ = xyz_interp.T @ np.diag(illuminant_interp) @ visible_range_data.T # shape (3, m*n)
    print("- Reflectance data converted to XYZ tristimulus values.")

    # Normalize the XYZ values to fit into the sRGB range
    XYZ_normalized = exposure.rescale_intensity(XYZ, out_range=(0, 1))
    print("- XYZ tristimulus values normalized to fit into the sRGB range.")

    # XYZ to sRGB
    XYZ_image = XYZ_normalized.T.reshape(nrows, ncols, 3)
    SRGB_image = colour.XYZ_to_sRGB(XYZ_image)
    print("- XYZ tristimulus values converted to sRGB.")

    # Notice that the sRGB values converted from XYZ could be smaller than 0 and larger than 1,
    # which are generally considered out-of-gamut or not physically meaningful for display purposes.
    # So we need to properly clip and stretch the sRGB values to preserve colors as much as possible for display.
    SRGB_image = np.clip(SRGB_image, 0, 1)
    print("- sRGB values clipped to fit into the displayable range.")

    # Stretch the sRGB image to enhance the contrast
    SRGB_image_percet_stretch = percentile_stretching(SRGB_image, stretch_percent)
    print(f"- Percentile stretching applied to the sRGB image. (%{stretch_percent})")

    # Apply white balance to the sRGB image
    SRGB_image_wb = white_balance(SRGB_image_percet_stretch)
    print("- White balance applied to the sRGB image.")

    # Apply the contrast stretch (if needed)
    SRGB_clahe_on_L = skimage_clahe_for_color_image(SRGB_image_wb)
    print("- CLAHE applied to the sRGB image.")
    display_images = (SRGB_image, SRGB_image_percet_stretch, SRGB_image_wb, SRGB_clahe_on_L)
    subplot_titles = ['sRGB', f'sRGB ({stretch_percent}% stretch)', 'sRGB (stretch+wb)', 'sRGB (stretch+wb+CLAHE)']
    if visualize:
        savefolder = input_path.parent / 'outputs' if savefolder is None else savefolder
        make_compare_plots(images=display_images,
                        suptitle=f'Visualization_from_colorimetric_conversion {illuminant}',
                        subplot_titles=subplot_titles,
                        saveimages=saveimages,
                        savefolder=savefolder)
    
    return display_images

    

if __name__ == "__main__":
    input_folder = Path("/home/fzhcis/mylab/data/rit-cis-hyperspectral-Symeon/data")
    infile_base_name = "Symeon_VNIR_cropped"

    # input_folder = Path("/home/fzhcis/mylab/gdrive/projects_with_Dave/for_Fei/Data/Ducky_and_Fragment")
    # infile_base_name = "fragment_cropped_FullSpec_2"

    # input_folder = Path("/home/fzhcis/mylab/data/HeiPorSPECTRAL_example/data/subjects/P086/2021_04_15_09_22_02")
    # input_path = input_folder / "2021_04_15_09_22_02_SpecCube.dat"

    # input_folder = Path("/home/fzhcis/mylab/data/dave-multispectral-truecolorhsi-whitebalance")
    # infile_base_name = "MSS_11_UR_35v_DataCube"
    input_path = input_folder / (infile_base_name + ".hdr")

    output_folder = Path("examples") / input_path.stem
    output_folder.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {output_folder}")
    visualize = True
    saveimages = True
    illuminant = 'D65' # choose from 'D50', 'D55', 'D65', 'D75'

    vanilla_display_images = vanilla_visualization(input_path, visualize=visualize, saveimages=saveimages, savefolder=output_folder)
    colorimetric_display_images = colorimetric_visualization(input_path, illuminant, visualize=visualize, saveimages=saveimages, savefolder=output_folder)