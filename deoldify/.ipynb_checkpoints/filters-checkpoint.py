from numpy import ndarray
from abc import ABC, abstractmethod
from .critics import colorize_crit_learner
from fastai.core import *
from fastai.vision import *
from fastai.vision.image import *
from fastai.vision.data import *
from fastai import *
import math
from scipy import misc
import cv2
from PIL import Image as PilImage

import torchvision.transforms


class IFilter(ABC):
    @abstractmethod
    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int
    ) -> PilImage:
        pass


class BaseFilter(IFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__()
        
        self.learn = learn.to_fp16()
        self.device = next(self.learn.model.parameters()).device
        self.norm, self.denorm = normalize_funcs(*stats)

    def _transform(self, image: PilImage) -> PilImage:
        return image
    def _scale_to_square(self, orig: PilImage, targ: int) -> PilImage:
        # a simple stretch to fit a square really makes a big difference in rendering quality/consistency.
        
        x, y = orig.size
        
        if (x == targ):
            return orig
        else:    
            targ_sz = (targ, targ)
            return orig.resize(targ_sz, resample=PIL.Image.BILINEAR)

    def _get_model_ready_image(self, orig: PilImage, sz: int) -> PilImage:
        result = self._scale_to_square(orig, sz)
        result = self._transform(result)
        return result

    def _model_process(self, orig: PilImage, sz: int) -> PilImage:
        model_image = self._get_model_ready_image(orig, sz)
        x = pil2tensor(model_image, np.float32)
        x = x.to(self.device)
        x.div_(255)
        x, y = self.norm((x, x), do_x=True)
        
        try:
            x, y = x.half(), y.half() #convert to float16
            result = self.learn.pred_batch(
                ds_type=DatasetType.Valid, batch=(x[None], y[None]), reconstruct=True
            )
        except RuntimeError as rerr:
            if 'memory' not in str(rerr):
                raise rerr
            print('Warning: render_factor was set too high, and out of memory error resulted. Returning original image.')
            return model_image
            
        out = result[0]
        out = out
        out = self.denorm(out.px, do_x=False)
        out = image2np(out * 255).astype(np.uint8)
        return PilImage.fromarray(out)

    def _unsquare(self, image: PilImage, orig: PilImage) -> PilImage:
        targ_sz = orig.size
        image = image.resize(targ_sz, resample=PIL.Image.BILINEAR)
        return image


class ColorizerFilter(BaseFilter):
    def __init__(self, learn: Learner, stats: tuple = imagenet_stats):
        super().__init__(learn=learn.to_fp16(), stats=stats)
        self.render_base = 10 #16

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int, post_process: bool = True) -> PilImage:
        render_sz = render_factor * self.render_base
        model_image = self._model_process(orig=filtered_image, sz=render_sz)
        raw_color = self._unsquare(model_image, orig_image)

        if post_process:
            return self._post_process(raw_color, orig_image)
        else:
            return raw_color

    def _transform(self, image: PilImage) -> PilImage:
        return image.convert('LA').convert('RGB')

    def _post_process(self, raw_color: PilImage, orig: PilImage) -> PilImage:
        color_np = np.asarray(raw_color)
        orig_np = np.asarray(orig)
        color_yuv = cv2.cvtColor(color_np, cv2.COLOR_BGR2YUV)
        # do a black and white transform first to get better luminance values
        orig_yuv = cv2.cvtColor(orig_np, cv2.COLOR_BGR2YUV)
        hires = np.copy(orig_yuv)
        hires[:, :, 1:3] = color_yuv[:, :, 1:3]
        final = cv2.cvtColor(hires, cv2.COLOR_YUV2BGR)
        final = PilImage.fromarray(final)
        return final
    
class MasterFilter(BaseFilter):
    def __init__(self, filters: [IFilter], render_factor: int):
        self.filters = filters
        self.render_factor = render_factor

    def filter(
        self, orig_image: PilImage, filtered_image: PilImage, render_factor: int = None, post_process: bool = True, patch_size:int=256) -> PilImage:
        render_factor = self.render_factor if render_factor is None else render_factor
        for filter in self.filters:
            filtered_image = filter.filter(orig_image, filtered_image, render_factor, post_process)
                
        return filtered_image

# was used to preprocess using MIRNet, but no longer used (used Wan et al.'s model insteads)
"""
class ModelFilter(IFilter):
    def __init__(self, model: nn.Module = MIRNet(), weights_path: str = './pretrained_models/denoising/model_denoising.pth', 
                 patch_size: int=256):
        super().__init__()
        self.model = model
        utils.load_checkpoint(model, weights_path)
        self.patch_size = patch_size
        model.cuda()
        model.eval()
    
    def get_n_row_cols_(self, orig_image: PilImage, patch_size:int) -> (int, int):
        x, y = orig_image.size
        n_row = y // patch_size
        n_col = x // patch_size
        if n_row==0: n_row = 1
        if n_col==0: n_col = 1
        return n_row, n_col
    
    def get_image_patches(self, orig_image: PilImage, n_row:int, n_col:int):
        x, y = orig_image.size
        row_size = y // n_row
        col_size = x // n_col

        np_image = np.asarray(orig_image)
        patches = []
        for j in range(n_row):
            for i in range(n_col):
                start_x = i * col_size
                end_x = (i+1) * col_size
                start_y = j * row_size
                end_y = (j+1) * row_size

                # because of the remainder from //, need to differentiate for patch in last col and last row
                if (i == n_col-1): # last col
                    if (j == n_row-1): # last row
                        patches.append(np_image[start_y:, start_x:, :])
                    patches.append(np_image[start_y:end_y, start_x:, :])
                elif (j == n_row-1):
                    patches.append(np_image[start_y:, start_x:end_x, :])
                else:
                    patches.append(np_image[start_y:end_y, start_x:end_x, :])

        return patches
    
    def make_PilImage(self, patches:[np.array]) -> [PilImage]:
        patches_pil = []
        for patch in patches:
            patches_pil.append(PilImage.fromarray(patch))

        return patches_pil

    def make_npArray(self, patches:[PilImage]) -> [np.array]:
        patches_np = []
        for patch in patches:
            patches_np.append(np.asarray(patch))

        return patches_np

    def make_TensorFromPilImage(self, patches:[PilImage]) -> [Tensor]:
        patches_t = []
        for patch in patches:
            patches_t.append(torchvision.transforms.ToTensor()(patch).float())

        return patches_t

    def make_PilImageFromTensor(self, patches:[Tensor]) -> [PilImage]:
        patches_pil = []
        for patch in patches:
            patches_pil.append(torchvision.transforms.ToPILImage()(patch))

        return patches_pil
        
    def squarify_patches(self, im_patches: [PilImage], patch_size:int) -> [PilImage]:
        sq_patches = []
        for patch in im_patches:
            x, y = patch.size
            targ_sz = (patch_size, patch_size)

            sq_patches.append(patch.resize(targ_sz, resample=PilImage.BILINEAR))

        return sq_patches

    def unsquare_patches(self, orig_patches: [PilImage], square_patches:[PilImage]) -> [PilImage]:

        unsq_patches = []
        for i in range(len(square_patches)):
            unsq_patches.append(square_patches[i].resize(orig_patches[i].size, resample=PilImage.BILINEAR))
        return unsq_patches

    def assemble_patches(self, patches:[np.array], n_col:int, n_row:int) -> [np.array]:
        idx = 0
        concated_row = []
        for i in range(n_row):
            concated_row.append(np.concatenate((patches[idx:idx+n_col]), axis=1))
            idx += n_col

        assembled = []
        assembled.append(np.concatenate((concated_row), axis=0))
        plt.imshow(assembled[0])
        plt.show()
        return assembled
    
    def display(self, ims:[PilImage], n_row, n_col):
        fig = plt.figure(figsize=(15,15))
        for i in range(len(ims)-1):
            ax = fig.add_subplot(n_col, n_row, i+1)
            ax.set_title(ims[i].size)
            imgplot = plt.imshow(ims[i])
        plt.show()
        
    def denoise(self, patches: [Tensor], model: nn.Module):
    
        patches_denoised = []
        with torch.no_grad():
            for patch in patches:
                patch = torch.unsqueeze(patch, 0)
                patch = patch.cuda()
                patches_denoised.append(torch.squeeze(model(patch), 0).cpu().detach())

        return patches_denoised
    
    def filter(self, orig_image: PilImage, filtered_image: PilImage, render_factor, post_process, patch_size: int=256) -> PilImage:
        
        n_row, n_col = self.get_n_row_cols_(filtered_image, self.patch_size)
        im_patches = self.get_image_patches(filtered_image, n_row, n_col)
        pil_ims = self.make_PilImage(im_patches)
        squared_pil_ims = self.squarify_patches(pil_ims, self.patch_size)
        squared_tensors = self.make_TensorFromPilImage(squared_pil_ims)
        den = self.denoise(squared_tensors, self.model)
        den = self.make_PilImageFromTensor(den)
        
        unsq = self.unsquare_patches(pil_ims, den)
        unsq = self.make_npArray(unsq)
        ass = self.assemble_patches(unsq, n_col, n_row)
        print(ass[0].shape," np")
        ass = self.make_PilImage(ass)[0]
        print(ass.size, "pil")
        return ass
"""