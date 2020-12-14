import fastai
from fastai import *
from fastai.core import *
from fastai.vision.transform import get_transforms
from fastai.vision.data import ImageImageList, ImageDataBunch, imagenet_stats
from .augs import noisify

# Function modified to allow use of custom samplers
def get_colorize_data_with_samplers(sz:int, bs:int, crappy_path:Path, good_path:Path, random_seed:int=None,
        keep_pct:float=1.0, num_workers:int=4, samplers=None, valid_pct=0.2, xtra_tfms=[])->ImageDataBunch:
    src = (ImageImageList.from_folder(crappy_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(valid_pct, seed=random_seed))
    if xtra_tfms is not None:
        data = (src.label_from_func(lambda x: good_path/x.relative_to(crappy_path))
        .transform((xtra_tfms,None), size=sz, tfm_y=True)
        .databunch(bs=bs, num_workers=num_workers, sampler=samplers, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    else:
        data = (src.label_from_func(lambda x: good_path/x.relative_to(crappy_path))
        .transform(get_transforms(max_zoom=1.5, max_lighting=0.4, max_warp=0.25, xtra_tfms=[]), size=sz, tfm_y=True)
        .databunch(bs=bs, num_workers=num_workers, sampler=samplers, no_check=True)
        .normalize(imagenet_stats, do_y=True))
    data.c = 3
    return data


def get_colorize_data(
    sz: int,
    bs: int,
    crappy_path: Path,
    good_path: Path,
    random_seed: int = None,
    keep_pct: float = 1.0,
    num_workers: int = 8,
    stats: tuple = imagenet_stats,
    xtra_tfms=[],
) -> ImageDataBunch:

    src = (
        ImageImageList.from_folder(crappy_path, convert_mode='RGB')
        .use_partial_data(sample_pct=keep_pct, seed=random_seed)
        .split_by_rand_pct(0.1, seed=random_seed)
    )

    data = (
        src.label_from_func(lambda x: good_path / x.relative_to(crappy_path))
        # apply xtra_tfms only on training set
        .transform(
            xtra_tfms,
            size=sz,
            tfm_y=False,
        )
        .databunch(bs=bs, num_workers=num_workers, no_check=True)
        .normalize(stats, do_y=True)
    )

    data.c = 3
    return data


def get_dummy_databunch() -> ImageDataBunch:
    path = Path('./dummy/')
    return get_colorize_data(
        sz=1, bs=1, crappy_path=path, good_path=path, keep_pct=0.001
    )

