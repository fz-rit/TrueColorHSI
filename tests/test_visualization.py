from pathlib import Path
from truecolorhsi.visualization import vanilla_visualization, colorimetric_visualization


# input_folder = Path("/home/fzhcis/mylab/data/rit-cis-hyperspectral-Symeon/data")
# infile_base_name = "Symeon_VNIR_cropped"
input_folder = Path("/home/fzhcis/mylab/gdrive/projects_with_Dave/for_Fei/Data/Ducky_and_Fragment")
infile_base_name = "fragment_cropped_FullSpec_2"
header_file = input_folder / (infile_base_name + ".hdr")
output_folder = Path("examples/images")
visualize = True
saveimages = False
illuminant = 'D65' # choose from 'D50', 'D55', 'D65', 'D75'

vanilla_display_images = vanilla_visualization(header_file, visualize=visualize, saveimages=saveimages, savefolder=output_folder)
colorimetric_display_images = colorimetric_visualization(header_file, illuminant, visualize=visualize, saveimages=saveimages, savefolder=output_folder)