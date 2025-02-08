from pathlib import Path
from truecolorhsi.visualization import vanilla_visualization, colorimetric_visualization

input_folder = Path("/home/fzhcis/mylab/data/rit-cis-hyperspectral-Symeon/data")
infile_base_name = "Symeon_VNIR_cropped"

# input_folder = Path("/home/fzhcis/mylab/gdrive/projects_with_Dave/for_Fei/Data/Ducky_and_Fragment")
# infile_base_name = "fragment_cropped_FullSpec_2"

# input_folder = Path("/home/fzhcis/mylab/data/HeiPorSPECTRAL_example/data/subjects/P086/2021_04_15_09_22_02")
# header_file = input_folder / "2021_04_15_09_22_02_SpecCube.dat"

# input_folder = Path("/home/fzhcis/mylab/data/dave-multispectral-truecolorhsi-whitebalance")
# infile_base_name = "MSS_11_UR_35v_DataCube"
header_file = input_folder / (infile_base_name + ".hdr")

output_folder = Path("examples") / header_file.stem
visualize = True
saveimages = False
illuminant = 'D65' # choose from 'D50', 'D55', 'D65', 'D75'

vanilla_display_images = vanilla_visualization(header_file, visualize=visualize, saveimages=saveimages, savefolder=output_folder)
colorimetric_display_images = colorimetric_visualization(header_file, illuminant, visualize=visualize, saveimages=saveimages, savefolder=output_folder)