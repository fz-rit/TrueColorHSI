# -*- coding: utf-8 -*-
"""
+
=======================================================================

 NAME:
      true_color_spectral_image_white

 DESCRIPTION:
Produce a true color image from a spectral image using the methods from M. Maali-Amiri, et al (2024) but including white balancing.  Based on new code by Fei Zhang.  Includes selection of pixels on a white target to the white balance.

 USES:


 PARAMETERS:


 KEYWORDS:


 RETURNS:


 NOTES:

 HISTORY:
01/17/2025: D. Messinger - created

@author: Morteza, David Messenger, Fei Zhang
"""

import spectral
import cv2
import numpy as np
import matplotlib.pyplot as plt
import colour
import skimage.exposure as exposure
from scipy.interpolate import interp1d
from truecolorhsi.accessories import get_illuminant_spd_and_xyz
from pathlib import Path
import skimage
import sys
import truecolorhsi.WBsRGB as wb_srgb

###########################
#
# show a single band image and collect pixel values by clicking with right mouse button
# input image is single band, uint 8 image
#
###########################
def collect_points_from_image(image2show):

# create the figures for mouse clicking
    fig, ax1  = plt.subplots()
    plt.suptitle("Use the right mouse button to pick points; at least 5 in the white target. \n"
             "Select no points to terminate and start over.\n""Close the figure when finished.")

    ax1.imshow(image2show, cmap = 'gray')

# initialize points
    selected_points = []

###
# define the click events, for right mouse button
###
    def on_click_white(event):
        if event.inaxes == ax1 and event.button == 3:  # right mouse button clicked in image
            selected_points.append((event.xdata, event.ydata))
            ax1.plot(event.xdata, event.ydata, 'ro')  # Plot a red dot at the clicked point
            ax1.figure.canvas.draw_idle()

# Connect the mouse click events to the respective axes
    fig.canvas.mpl_connect('button_press_event', on_click_white)
    plt.show()

# convert the points lists to arrays of numbers
    selected_points = np.array(selected_points)
# and then convert to integers for indexing later on
    selected_points = selected_points.astype(int)
    npts_selected = len(selected_points)
    # print('# of points selected: ', npts_selected)
    if npts_selected == 0 :
        sys.exit('No points selected.  Stopping....')

    return(selected_points)
##################


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

def white_balance(image: np.ndarray):
    """
    White balance the input image using the WBsRGB method.
    Ref: https://github.com/mahmoudnafifi/WB_sRGB/tree/master

    Parameters:
    image: the input image

    Returns:
    outImg: the white balanced image
    """

    wbModel = wb_srgb.WBsRGB(gamut_mapping=2,
                            upgraded=1)
    outImg = wbModel.correctImage(image)  # white balance it

    return outImg

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

    viz_norm_white = white_balance(viz_norm)

    # Apply more advanced contrast stretch: CLAHE (Contrast Limited Adaptive Histogram Equalization)
    viz_clahe_on_L = skimage_clahe_for_color_image(viz_norm_white)

    display_images = [viz_norm, viz_norm_white]
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

    # # # Apply the contrast stretch (if needed)
    # # SRGB_clahe_on_L = skimage_clahe_for_color_image(SRGB_image)
    # # display_images = [SRGB_image, SRGB_clahe_on_L]
    # # make_compare_plots(images=display_images,
    # #                     suptitle='Visualization_from_colorimetric_conversion',
    # #                     subplot_title=f'{illuminant}-based sRGB',
    # #                     saveimages=saveimages,
    # #                     savefolder=savefolder)

    # # plt.show()

    # #############
    # #  Perform white balancing by selecting pixels on the white calibration target and force them to be "white"
    # #############

    # # choose a band to display
    # # band2show = get_band_index(bands, 550.0)
    # #
    # # print('band2show: ', band2show)

    # # convert the chosen image to uint8 for openCV
    # # image_uint8 = hyperspectral_data[:, :, band2show].astype('uint8')

    # image_uint8 = SRGB_image[:,:,0]*255.0
    # image_uint8 = image_uint8.astype('uint8')

    # # call the function
    # white_points = collect_points_from_image(image_uint8)
    # # determine how many of each were chosen
    # npts_white = len(white_points)

    # print('# of points: ', npts_white)

    # if npts_white == 0:
    #     sys.exit('No points selected.  Stopping....')
    # # -->>  have to reverse the points based on how they are returned by the mouse clicks
    # # compute the average white spectrum
    # white_point_avg = np.zeros(3)
    # for i, j in white_points:
    #     white_point_avg = white_point_avg + SRGB_image[j, i, :]
    # white_point_avg = white_point_avg / npts_white

    # print('white_point avg: ', white_point_avg)

    # # define "true white" as [243 243 242]/255
    # W = np.array([243.0, 243.0, 242.0])
    # W = W / 255.0
    # # print('W: ', W)
    # # sys.exit()
    # # compute the scale factor
    # X = W / white_point_avg

    # print('scale factor X: ', X)
    # SRGB_image_white = np.zeros((nrows,ncols,3))
    # SRGB_image_white[:,:,0] = SRGB_image[:,:,0] * X[0]
    # SRGB_image_white[:,:,1] = SRGB_image[:,:,1] * X[1]
    # SRGB_image_white[:,:,2] = SRGB_image[:,:,2] * X[2]

    SRGB_image_white = white_balance(SRGB_image)
    display_images = [SRGB_image, SRGB_image_white]
    make_compare_plots(images=display_images,
                        suptitle='Visualization_from_colorimetric_conversion with white balance',
                        subplot_title=f'{illuminant}-based sRGB',
                        saveimages=saveimages,
                        savefolder=savefolder)

    plt.show()


if __name__ == "__main__":
    # Specify the folder path containing the ENVI files
    # input_folder = Path("/Users/dwmpaci/Desktop/3_PROJECTS/1_NEH_MISHA/2025_01_07_URochester/")
    input_folder = Path("/home/fzhcis/mylab/data/dave-multispectral-truecolorhsi-whitebalance")
    infile_base_name = "MSS_11_UR_35v_DataCube"
    output_folder = Path("examples/multispec_images")
    saveimages = True
    illuminant = 'D65' # choose from 'D50', 'D55', 'D65', 'D75'
    # Read the hyperspectral image using spectral
    header_file = input_folder / (infile_base_name + ".hdr")
    spectral_image = spectral.open_image(header_file)
    hyperspectral_data = spectral_image.load()

    # nrows = hyperspectral_data.nrows
    # ncols = hyperspectral_data.ncols
    # nbands = hyperspectral_data.nbands
    # print('IMAGE rows, cols, bands: ', hyperspectral_data.nrows, hyperspectral_data.ncols, hyperspectral_data.nbands)
    # print('')
    # print('wavelengths: ', hyperspectral_data.bands.centers)

    vanilla_visualization(hyperspectral_data, saveimages, output_folder)
    colorimetric_visualization(hyperspectral_data, illuminant, saveimages, output_folder)

