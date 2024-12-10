import numpy as np
import colour
from colour.colorimetry import sd_to_XYZ
import matplotlib.pyplot as plt


def get_illuminant_spd_and_xyz(illuminant: str = 'D65', 
                    verbose: bool = False, 
                    plot_flag: bool = False, 
                    run_example: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    """
    Get the D65 illuminant and the CIE 1931 2° standard observer color matching functions.

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
        illuminant_spd = colour.SDS_ILLUMINANTS['D50'] # Image will look reddish
    elif illuminant == 'D65':
        illuminant_spd = colour.SDS_ILLUMINANTS['D65'] # Ideal natural daylight, so hopefully the best
    elif illuminant == 'D75':
        illuminant_spd = colour.SDS_ILLUMINANTS['D75'] # Image will look bluish
    else:
        raise ValueError("Invalid illuminant. Choose from 'D50', 'D65', or 'D75'.")
    

    # Get the CIE 1931 2° standard observer color matching functions
    cmfs = colour.MSDS_CMFS['CIE 1931 2 Degree Standard Observer']

    if verbose:
        print("D65 Illuminant SPD:", illuminant_spd.values)
        print("D65 Illuminant Wavelengths:", illuminant_spd.wavelengths)
        print("CIE 1931 2° standard observer values:", cmfs.values)
        print("CIE 1931 2° standard observer wavelengths:", cmfs.wavelengths)


    # Align the shape of illuminant to the CMFs,
    # since the CMFs has better granularity
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
        # Plot the D65 illuminant SPD and the CIE 1931 2° standard observer color matching functions
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Plot D65 Illuminant SPD
        ax1.plot(wavelengths, illuminant_spd_values, label='D65 SPD')
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Relative Power')
        ax1.set_title('D65 Illuminant Spectral Power Distribution')
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
        # Calculate the XYZ tristimulus values of the D65 illuminant
        XYZ = sd_to_XYZ(illuminant_spd, cmfs=cmfs)
        # Normalize to fit into RGB range
        XYZ_normalized = XYZ / max(XYZ)  
        # Convert to sRGB
        RGB_display = colour.XYZ_to_sRGB(XYZ_normalized) * 255
        print("Displayable RGB (8-bit):", RGB_display)
        print("XYZ Tristimulus Values of D65 Illuminant:")
        print(XYZ)
        print("Normalized XYZ Tristimulus Values of D65 Illuminant:")
        print(XYZ_normalized)
        print("RGB Values of D65 Illuminant:")
        print(RGB_display)

    return wavelengths, illuminant_spd_values, xyz

if __name__ == "__main__":
    wavelengths, illuminant_spd_values, xyz = get_illuminant_spd_and_xyz(illuminant='D65', 
                                                            verbose=False, # flip it to True to checkout more details.
                                                            plot_flag=True, 
                                                            run_example=True)
