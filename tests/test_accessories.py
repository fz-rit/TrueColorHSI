from truecolorhsi.accessories import get_illuminant_spd_and_xyz

wavelengths, illuminant_spd_values, xyz = get_illuminant_spd_and_xyz(illuminant='D65', 
                                                            verbose=True, # flip it to True to checkout more details.
                                                            plot_flag=False, 
                                                            run_example=True)