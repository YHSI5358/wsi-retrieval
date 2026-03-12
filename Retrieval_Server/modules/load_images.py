import sys, os, logging
from tqdm import tqdm
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
from PIL import Image
from openslide import OpenSlide

logger = logging.getLogger(__name__)

def load_svsortiff_image(img_path, open_image=True):
    cache_path = "Retrieval_Server/caches/patch_image"
    os.makedirs(cache_path, exist_ok=True)
    file_list = os.listdir(cache_path)

    slide = OpenSlide(img_path) 
    slice_size = (2048, 2048)
    num_level = slide.level_count

    related_pathes = [img_path]
    positions = [(-1, -1)]
    levels = [-1]

    for level in range(num_level):
        width, height = slide.level_dimensions[level]
        downsample = int(slide.level_downsamples[level])

        logging.info(f"Getting WSI Patch image of {os.path.basename(img_path)} in level: {level}.")

        for x in range(0, width, slice_size[0]):
            for y in range(0, height, slice_size[1]):
                save_name = f"{os.path.basename(img_path).split(".")[0]}_{level}_{x}_{y}.png"
                save_path = os.path.join(cache_path, save_name)
                if save_name not in file_list and open_image:
                    image = slide.read_region((x*downsample, y*downsample), level, slice_size)
                    image = image.convert('RGB') 
                    image.save(save_path)
                related_pathes.append(save_path)
                positions.append((x, y))
                levels.append(level)
    slide.close()

    return related_pathes, positions, levels

def load_pngorjpg_image(img_path):
    image = Image.open(img_path)
    # resize_img = image.resize((2048, 2048))

    return image


if __name__ == "__main__":
    svs_file = 'Retrieval_Server/caches/wsi_image/IBL012-7-12.ibl.tiff'
    related_pathes, positions = load_svsortiff_image(svs_file)
    for path in related_pathes:
        print(path)
    # svs_image.save("Retrieval_Server/caches/"+svs_file.split("/")[-1].split(".")[0]+".png")

    # tiff_file = '/home/mdi/suri/MDI_RAG_project/Retrieval_Server/data/images/IBL014-7-14.ibl.tiff'
    # # target_size = (256, 256)  #  
    # related_pathes, positions = load_svsortiff_image(tiff_file)
    # for path in related_pathes:
    #     print(path)
    # tiff_image.save("Retrieval_Server/caches/"+tiff_file.split("/")[-1].split(".")[0]+".png")