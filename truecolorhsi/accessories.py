import numpy as np
import colour
from colour.colorimetry import sd_to_XYZ
import matplotlib.pyplot as plt
from pathlib import Path
import spectral
from typing import Union
from skimage import exposure
import truecolorhsi.WBsRGB as wb_srgb
# import WBsRGB as wb_srgb


def get_illuminant_spd_and_xyz(illuminant: str = 'D65', 
                    verbose: bool = False, 
                    plot_flag: bool = False, 
                    run_example: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Get the illuminant (D50, D65, or D75) and the CIE 1931 2° standard observer color matching functions.

    Parameters:
    verbose: If True, print the illuminant and color matching functions values and wavelengths.
    plot_flag: If True, plot the illuminant and color matching functions.

    Returns:
    wavelengths: Wavelengths of the illuminant and color matching functions.
    illuminant_spd_values: Values of the D65 illuminant.
    xyz: Color matching functions values.

    """

    # Get the spectral power distribution of illuminant D50, D65, or D75
    if illuminant == 'D50':
        illuminant_spd = colour.SDS_ILLUMINANTS['D50'] # Image will look yellowish or reddish
    elif illuminant == 'D55':
        illuminant_spd = colour.SDS_ILLUMINANTS['D55'] # Image will look yellowish or reddish
    elif illuminant == 'D65':
        illuminant_spd = colour.SDS_ILLUMINANTS['D65'] # Ideal natural daylight, so hopefully the best
    elif illuminant == 'D75':
        illuminant_spd = colour.SDS_ILLUMINANTS['D75'] # Image will look bluish
    else:
        raise ValueError("Invalid illuminant. Choose from 'D50', 'D55', 'D65', or 'D75'.")
    

    # Get the CIE 1931 2° standard observer color matching functions
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

    if verbose:
        print(f"{illuminant} Illuminant SPD:", illuminant_spd.values)
        print(f"{illuminant} Illuminant Wavelengths:", illuminant_spd.wavelengths)
        print("CIE 1931 2° standard observer values:", cmfs.values)
        print("CIE 1931 2° standard observer wavelengths:", cmfs.wavelengths)


    # Align the shape of illuminant to the CMFs,
    # since the CMFs has better granularity: 
    # CMFs_wavelengths(360, 830, 1) vs illuminant_wavelenghths(300, 780, 5)
    illuminant_spd = illuminant_spd.copy().align(cmfs.shape)
    # Get wavelengths and values
    wavelengths = illuminant_spd.wavelengths
    illuminant_spd_values = illuminant_spd.values
    

    # CMFs values
    x_bar = cmfs.values[..., 0]
    y_bar = cmfs.values[..., 1]
    z_bar = cmfs.values[..., 2]

    # Combine CMFs into a single array if needed
    xyz = np.stack((x_bar, y_bar, z_bar), axis=-1)

    if plot_flag:
        # Plot the illuminant SPD and the CIE 1931 2° standard observer color matching functions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.plot(wavelengths, illuminant_spd_values, label='D65 SPD')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Relative Power')
        ax1.set_title(f"{illuminant}  Illuminant Spectral Power Distribution")
        ax1.legend()

        # Plot Color Matching Functions
        ax2.plot(wavelengths, x_bar, label='x̄(λ)', color='r')
        ax2.plot(wavelengths, y_bar, label='ȳ(λ)', color='g')
        ax2.plot(wavelengths, z_bar, label='z̄(λ)', color='b')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title('CIE 1931 2° Standard Observer Color Matching Functions')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    if run_example:
        # Calculate the XYZ tristimulus values of the illuminant
        XYZ = sd_to_XYZ(illuminant_spd, cmfs=cmfs)
        # Normalize to fit into RGB range
        XYZ_normalized = XYZ / max(XYZ)  
        # Convert to sRGB
        RGB_display = colour.XYZ_to_sRGB(XYZ_normalized) * 255
        print("Displayable RGB (8-bit):", RGB_display)
        print(f"XYZ Tristimulus Values of {illuminant} Illuminant:")
        print(XYZ)
        print(f"Normalized XYZ Tristimulus Values of {illuminant} Illuminant:")
        print(XYZ_normalized)
        print(f"RGB Values of {illuminant} Illuminant:")
        print(RGB_display)

    return wavelengths, illuminant_spd_values, xyz


def read_envi_hsi(header_file: Union[str, Path]) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the hyperspectral image using spectral.

    Parameters:
        header_file: The header file of the hyperspectral image.
    
    Returns:
        hyperspectral_cube: The hyperspectral data cube.
        band_centers: The band centers of the hyperspectral data.

    """
    header = Path(header_file) if isinstance(header_file, str) else header_file
    spectral_image = spectral.open_image(header)
    hyperspectral_data = spectral_image.load() 
    band_centers = spectral_image.bands.centers
    hyperspectral_cube = np.array(hyperspectral_data)
    return hyperspectral_cube, band_centers


def read_tivita_hsi(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Tivita biomedical data (the Heidelberg Porcine HyperSPECTRAL Imaging Dataset) cube as Numpy array.

    Ref: https://github.com/IMSY-DKFZ/htc/blob/bd252fcf2a2065e3c28567eaa2f892f2908eb0b8/htc/tivita/hsi.py#L18

    Args:
        path: Path to the cube file (e.g. "[...]/2020_07_20_18_17_26/2020_07_20_18_17_26_SpecCube.dat").

    Returns:
        cube: The hyperspectral data cube.
        band_centers: The band centers of the hyperspectral data.
    """
    assert path.exists() and path.is_file(), f"Data cube {path} does not exist or is not a file"

    shape = np.fromfile(path, dtype=">i", count=3)  # Read shape of HSI cube
    cube = np.fromfile(
        path, dtype=">f", offset=12
    )  # Read 1D array in big-endian binary format and ignore first 12 bytes which encode the shape
    cube = cube.reshape(*shape)  # Reshape to data cube
    cube = np.flip(cube, axis=1)  # Flip y-axis to match RGB image coordinates

    cube = np.swapaxes(cube, 0, 1)  # Consistent image shape (height, width)
    cube = cube.astype(np.float32)  # Consistently convert to little-endian

    # band_centers information are from https://heiporspectral.org/
    band_centers = np.arange(500, 1000, 5)  # 100 bands from 500 to 1000 nm
    return cube, band_centers

def read_HSI_data(input_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the hyperspectral data cube as Numpy array.

    Args:
        input_path: Path to the hyperspectral data cube.
    
    Returns:
        hyperspectral_cube: The hyperspectral data cube.
        band_centers: The band centers of the hyperspectral data.
    """
    if input_path.suffix == ".hdr":
        hyperspectral_cube, band_centers = read_envi_hsi(input_path)
    elif input_path.suffix == ".dat":
        hyperspectral_cube, band_centers = read_tivita_hsi(input_path)
    else:
        raise ValueError("Invalid file format. Supported formats are .hdr and .dat.")
    
    return hyperspectral_cube, band_centers


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


def percentile_stretching(image: np.ndarray, percent: int=1) -> np.ndarray:
    """
    Perform percentile stretching on the given image.

    Args:
        image: Image to stretch.
        low: Lower percentile.
        high: Higher percentile.

    Returns: Stretched image.
    """
    assert 0 <= percent <= 10, f"Percentile must be between 0 and 10, but got {percent}"
    p_low, p_high = np.percentile(image, (percent, 100-percent))
    return exposure.rescale_intensity(image, in_range=(p_low, p_high))

def white_balance(image: np.ndarray, 
                     method: str = "ml_wb") -> np.ndarray:
    """
    White balance the input image using the WBsRGB method.
    Ref: https://github.com/mahmoudnafifi/WB_sRGB/tree/master

    Parameters:
        image: the input image
        gamut_mapping: 1: scale (prefered for oversaturated images), 2: clip
        upgraded: 0: the original WBsRGB, 1: the upgraded WBsRGB with more training data


    Returns:
    outImg: the white balanced image
    """
    if method == "ml_wb":
        wbModel = wb_srgb.WBsRGB(gamut_mapping=2,
                                upgraded=1)
        outImg = wbModel.correctImage(image)

        if np.min(outImg) < 0 or np.max(outImg) > 1:
            print("-- Warning: White balance output out of range [0, 1].")
            outImg = np.clip(outImg, 0, 1)
            print("-- Clipped to [0, 1].")

        print(f"WB Output range: [{np.min(outImg)}, {np.max(outImg)}]")
    elif method == "gray_world":
        classic_wb = wb_srgb.Classic_WB()
        outImg = classic_wb.gray_world_white_balance(image)
    elif method == "white_patch":
        classic_wb = wb_srgb.Classic_WB()
        outImg = classic_wb.white_patch_white_balance(image)
    else:
        raise ValueError(f"Invalid white balance method - {method}. Choose from 'ml_wb', 'gray_world', or 'white_patch'.")


    return outImg