import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("...")
import numpy as np
from multiprocessing import shared_memory
from openslide import OpenSlide
from torchvision import transforms
from torch.utils.data import Dataset



class WSIDataLoader():
    def wsi_loading_patch(self, wsi_path, position=(0,0), level=0, size=(256,256)):
        slide = OpenSlide(wsi_path)

        print(f":{wsi_path}")
        print(f"{slide.dimensions[0]} ")
        print(f"{slide.dimensions[1]} ")
        print(f" {slide.level_count}")
        print(f"{slide.level_dimensions}")
        print(f"{slide.level_downsamples}")

        ture_pos = tuple([int(pos*slide.level_downsamples[level]) for pos in position])
        image = slide.read_region(ture_pos, level, size)
        slide.close()
        image = image.convert('RGB')

        return image

    def wsi_loading_patches(self, wsi_path):
        """Get WSI infos and patches from WSI path"""
        slide = OpenSlide(wsi_path)
        levels, patch_size = slide.level_count, (256, 256)

        loaded_infos, loaded_images = [], []
        for level in range(1, slide.level_count):
            ratio = slide.level_downsamples[level]
            width, height = slide.level_dimensions[level][0], slide.level_dimensions[level][1]
            for w in range(0, width, patch_size[0]):
                for h in range(0, height, patch_size[0]):
                    ture_pos = (int(w * ratio), int(h * ratio))

                    infos = {
                        "wsi_name":wsi_path.split("/")[-1],
                        "position":ture_pos,    # basic on level 0
                        "level":level,
                        "size":patch_size,
                    }

                    image = slide.read_region(ture_pos, level, patch_size)
                    image = image.convert('RGB')
                    loaded_images.append(image)

        return loaded_infos, loaded_images 

    def get_wsi_shared_patches(self, wsi_path):
        """ loading patched image in share memory"""
        loaded_infos, loaded_images = self.wsi_loading_patches(wsi_path)
        img_arrays = np.array([np.array(img, dtype=np.uint8) for img in loaded_images])

        shm = shared_memory.SharedMemory(create=True, size=img_arrays.nbytes)
        shared_array = np.ndarray(img_arrays.shape, dtype=img_arrays.dtype, buffer=shm.buf)
        shared_array[:] = img_arrays[:]    

        return loaded_infos, shm, img_arrays.shape, img_arrays.dtype


if __name__ == "__main__":
    wsi_path = ""
    wsi_loader = WSIDataLoader()

    position, level, size = (0,0), 9, (200,200)
    wsi_patch_image = wsi_loader.wsi_loading_patch(wsi_path, position, level, size)
    wsi_patch_image.save("MDI_RAG_Image2Image_Research/data/cache/"+wsi_path.split("/")[-1].split(".")[0]+".png")

    position, level, size = (200,200), 8, (200, 200)
    wsi_patch_image = wsi_loader.wsi_loading_patch(wsi_path, position, level, size)
    wsi_patch_image.save("MDI_RAG_Image2Image_Research/data/cache/"+wsi_path.split("/")[-1].split(".")[0]+"1.png")