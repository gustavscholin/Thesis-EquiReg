import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def _get_light_image_augmenter(sq):
    aug = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=sq),
        iaa.Flipud(0.5, random_state=sq),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=sq),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=sq)
    ], random_order=True, random_state=sq)

    aug = aug.to_deterministic()
    return aug


def _get_light_seg_mask_augmenter(sq):
    aug = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=sq),
        iaa.Flipud(0.5, random_state=sq),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=sq, order=0),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=sq, order=0)
    ], random_order=True, random_state=sq)

    aug = aug.to_deterministic()
    return aug


def _get_image_augmenter(sq):
    shear_degrees = np.degrees(np.arctan(0.2))
    aug = iaa.SomeOf((3, None), [
        iaa.Fliplr(0.5, random_state=sq),
        iaa.Flipud(0.5, random_state=sq),
        iaa.Affine(rotate=(-20, 20), random_state=sq),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=sq),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=sq),
        iaa.Affine(shear=(-shear_degrees, shear_degrees), random_state=sq),
        iaa.Affine(scale=(0.9, 1.1), random_state=sq),
        # iaa.GammaContrast((0.8, 1.2), random_state=sq),
        iaa.ElasticTransformation(alpha=720, sigma=24, random_state=sq)
    ], random_order=True, random_state=sq)

    aug = aug.to_deterministic()
    return aug


def _get_seg_mask_augmenter(sq):
    shear_degrees = np.degrees(np.arctan(0.2))
    aug = iaa.SomeOf((3, None), [
        iaa.Fliplr(0.5, random_state=sq),
        iaa.Flipud(0.5, random_state=sq),
        iaa.Affine(rotate=(-20, 20), random_state=sq, order=0),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=sq, order=0),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=sq, order=0),
        iaa.Affine(shear=(-shear_degrees, shear_degrees), random_state=sq, order=0),
        iaa.Affine(scale=(0.9, 1.1), random_state=sq, order=0),
        # iaa.GammaContrast((0.8, 1.2), random_state=sq),
        iaa.ElasticTransformation(alpha=720, sigma=24, random_state=sq, order=0)
    ], random_order=True, random_state=sq)

    aug = aug.to_deterministic()
    return aug


def sup_aug(in_image, in_seg_mask):
    image = np.copy(in_image)
    seg_mask = np.copy(in_seg_mask.astype(np.int32))
    sq = np.random.SeedSequence()

    img_aug = _get_light_image_augmenter(sq)
    seg_mask_aug = _get_light_seg_mask_augmenter(sq)

    image = img_aug.augment(image=image)
    seg_mask = seg_mask_aug(image=seg_mask)

    return image, seg_mask

def unsup_img_aug(in_image):
    image = np.copy(in_image)
    sq = np.random.SeedSequence()
    aug = _get_image_augmenter(sq)
    image = aug.augment(image=image)

    entropy = np.array([sq.entropy], dtype=np.str)
    return image, entropy


def unsup_seg_aug(in_seg, entropy):
    entropy = int(entropy[0])

    in_seg = in_seg.astype(np.int32)
    seg = SegmentationMapsOnImage(np.copy(in_seg), in_seg.shape)
    aug = _get_image_augmenter(np.random.SeedSequence(entropy))
    seg_aug = aug.augment(segmentation_maps=seg)

    return seg_aug.get_arr().astype(np.int64)


def unsup_logits_aug(in_logits, entropy):
    out_logits = np.zeros(in_logits.shape, np.float32)
    for i in range(in_logits.shape[0]):
        seed_ent = int(entropy[i])

        logits = np.copy(in_logits[i, ...])
        aug = _get_seg_mask_augmenter(np.random.SeedSequence(seed_ent))
        logits_aug = aug.augment(image=logits)

        logits_aug[np.all(logits_aug == [0]*4, axis=-1)] = [1, 0, 0, 0]
        out_logits[i, ...] = logits_aug

    return out_logits

