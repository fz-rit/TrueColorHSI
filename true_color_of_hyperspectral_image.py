# -*- coding: utf-8 -*-
"""
External File:
- Accessories.py

@author: Morteza, David Messenger, Fei Zhang
"""

import spectral
import numpy as np
import matplotlib.pyplot as plt
import colour
import skimage.exposure as exposure
from scipy.interpolate import interp1d
from Accessories import get_illuminant_spd_and_xyz
from pathlib import Path
import skimage


def get_band_index(bandarray: np.ndarray, WL: float) -> int:
    """
    Get the index of the band closest to the specified wavelength in the bandarray,
    which was derived from hyperspectral_data.bands.centers.

    Parameters:
    bandarray: array of band center wavelengths
    WL: the wavelength of interest

    Returns:
    band_index: the index of the band closest to the specified wavelength.
    """

    nbands = np.size(bandarray)
    temp_array= np.ones(nbands) * WL
    band_index = np.argmin(np.abs(bandarray - temp_array))

    return band_index

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

    return equalized_image

def make_compare_plots(images: list, 
                       suptitle: str, 
                       subplot_title: str, 
                       saveimages: bool, 
                       savefolder: Path) -> None:
    """
    Make a comparison plot of the input images.

    Parameters:
    images: list of images to be compared
    suptitle: the title of the plot
    subplot_title: the title of each subplot
    saveimages: whether to save the plot as an image
    savefolder: the folder to save the image

    Returns:
    None
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    axes[0].imshow(images[0])
    axes[0].axis('off')
    axes[0].set_title(subplot_title)

    axes[1].imshow(images[1])
    axes[1].axis('off')
    axes[1].set_title(f'{subplot_title}(contrast enhanced with CLAHE)')
    fig.suptitle(suptitle, fontsize=16)
    fig.tight_layout()
    if saveimages:
        outfile = savefolder / f'{suptitle}.jpg'
        print('Writing to: ', outfile)
        plt.savefig(outfile, bbox_inches = 'tight', dpi = 300)

def vanilla_visualization(hyperspectral_data: spectral.image.ImageArray,
                        saveimages: bool,
                        savefolder: Path,) -> None:
    """
    Display the hyperspectral image by directly visualizing the RGB bands.

    Parameters:
    bands: array of band center wavelengths

    Returns:
    None
    """
    # Get the approximate r,g,b bands from the HSI
    bands = np.array(hyperspectral_data.bands.centers)
    iblue = get_band_index(bands,450.0)
    igreen = get_band_index(bands,550.0)
    ired = get_band_index(bands,650.0)

    # Vanilla visualization: Directly visualize the RGB bands
    viz_simple = hyperspectral_data[:,:, [ired, igreen, iblue]]

    # Normalize the data: rescale the intensity to [0, 1], which ends up contrast stretching the image.
    viz_norm = exposure.rescale_intensity(viz_simple)

    # Apply more advanced contrast stretch: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    viz_clahe_on_L = skimage_clahe_for_color_image(viz_norm)

    display_images = [viz_simple, viz_clahe_on_L]
    make_compare_plots(images=display_images, 
                       suptitle='Visualization_from_rgb_bands', 
                       subplot_title='RGB ',
                       saveimages=saveimages, 
                       savefolder=savefolder)




def colorimetric_visualization(hyperspectral_data: spectral.image.ImageArray, 
                               illuminant: str,
                               saveimages: bool, 
                               savefolder: Path) -> None:
    """
    Display the hyperspectral image by converting the reflectance data to sRGB using colorimetric methods.

    Parameters:
    hyperspectral_data: the hyperspectral image data
    saveimages: whether to save the plot as an image
    savefolder: the folder to save the image

    Returns:
    None

    """

    #Interpolating the standard data of standard illuminant and 
    #standard observer to coincide with the wavelengths that
    #our hyperspectral image has
    nrows, ncols, nbands = hyperspectral_data.shape
    print(f'IMAGE rows, cols, bands: {(nrows, ncols, nbands)}')
    bands = np.array(hyperspectral_data.bands.centers)
    i_cutoff = get_band_index(bands, 830.0)
    hyperspec_wavelengths = bands[:i_cutoff]

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
    visible_range_data = hyperspectral_data[:, :, :i_cutoff].reshape((-1, i_cutoff))

    # Convert Reflectance to CIEXYZ tristimulus values
    XYZ = xyz_interp.T @ np.diag(illuminant_interp) @ visible_range_data.T # shape (3, m*n)

    # Normalize the XYZ values to fit into the sRGB range
    XYZ_normalized = (XYZ - np.min(XYZ))  / (np.max(XYZ) - np.min(XYZ))

    # XYZ to sRGB
    XYZ_image = XYZ_normalized.T.reshape(nrows, ncols, 3)
    SRGB_image = colour.XYZ_to_sRGB(XYZ_image)

    # Notice that the sRGB values converted from XYZ could be smaller than 0 and larger than 1,
    # which are generally considered out-of-gamut or not physically meaningful for display purposes.
    # So we need to clip the sRGB values to preserve colors as much as possible for display.
    SRGB_image = SRGB_image.clip(0, 1) 

    # # Normalize the data (if needed)
    # SRGB_norm = exposure.rescale_intensity(SRGB_image) 

    # Apply the contrast stretch (if needed)
    SRGB_clahe_on_L = skimage_clahe_for_color_image(SRGB_image)
    display_images = [SRGB_image, SRGB_clahe_on_L]
    make_compare_plots(images=display_images,
                        suptitle='Visualization_from_colorimetric_conversion',
                        subplot_title=f'{illuminant}-based sRGB',
                        saveimages=saveimages,
                        savefolder=savefolder)

    plt.show()

    

if __name__ == "__main__":
    # Specify the folder path containing the ENVI files
    input_folder = Path("path/to/Symeon")
    infile_base_name = "Symeon_VNIR_cropped"
    output_folder = Path("outputs")
    saveimages = False
    illuminant = 'D75' # choose from 'D50', 'D55', 'D65', 'D75'
    # Read the hyperspectral image using spectral
    header_file = input_folder / (infile_base_name + ".hdr")
    spectral_image = spectral.open_image(header_file)
    hyperspectral_data = spectral_image.load() 

    vanilla_visualization(hyperspectral_data, saveimages, output_folder)
    colorimetric_visualization(hyperspectral_data, illuminant, saveimages, output_folder)