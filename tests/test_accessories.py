from truecolorhsi.accessories import get_illuminant_spd_and_xyz

wavelengths, illuminant_spd_values, xyz = get_illuminant_spd_and_xyz(illuminant='D65', 
                                                            verbose=True,
                                                            plot_flag=True, 
                                                            run_example=True)