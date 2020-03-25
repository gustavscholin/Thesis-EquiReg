"""
Module with augmenters.
"""
import numpy as np
import imgaug.augmenters as iaa


def _get_weak_image_augmenter(sq: np.random.SeedSequence = np.random.SeedSequence()) -> iaa.Augmenter:
    """
    Gets augmenter for weak augmentations of 2D images.
    :param sq: Numpy seed sequence
    :return: Imgaug augmenter
    """
    seeds = sq.generate_state(5)
    aug = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=seeds[0]),
        iaa.Flipud(0.5, random_state=seeds[1]),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=seeds[2]),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=seeds[3])
    ], random_order=True, random_state=seeds[4])

    aug = aug.to_deterministic()
    return aug


def _get_weak_seg_map_augmenter(sq: np.random.SeedSequence) -> iaa.Augmenter:
    """
    Gets augmenter for weak augmentations of 2D segmentation maps.
    :param sq: Numpy seed sequence
    :return: Imgaug augmenter
    """
    seeds = sq.generate_state(5)
    aug = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=seeds[0]),
        iaa.Flipud(0.5, random_state=seeds[1]),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=seeds[2], order=0),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=seeds[3], order=0)
    ], random_order=True, random_state=seeds[4])

    aug = aug.to_deterministic()
    return aug


def _get_strong_image_augmenter(sq: np.random.SeedSequence) -> iaa.Augmenter:
    """
    Gets augmenter for strong augmentations of 2D images.
    :param sq: Numpy seed sequence
    :return: Imgaug augmenter
    """
    seeds = sq.generate_state(9)
    shear_degrees = np.degrees(np.arctan(0.2))
    aug = iaa.SomeOf((3, None), [
        iaa.Fliplr(0.5, random_state=seeds[0]),
        iaa.Flipud(0.5, random_state=seeds[1]),
        iaa.Affine(rotate=(-20, 20), random_state=seeds[2]),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=seeds[3]),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=seeds[4]),
        iaa.Affine(shear=(-shear_degrees, shear_degrees), random_state=seeds[5]),
        iaa.Affine(scale=(0.9, 1.1), random_state=seeds[6]),
        iaa.ElasticTransformation(alpha=720, sigma=24, random_state=seeds[7])
    ], random_order=True, random_state=seeds[8])

    aug = aug.to_deterministic()
    return aug


def _get_strong_seg_map_augmenter(sq: np.random.SeedSequence) -> iaa.Augmenter:
    """
    Gets augmenter for strong augmentations of 2D segmentation maps.
    :param sq: Numpy seed sequence
    :return: Imgaug augmenter
    """
    seeds = sq.generate_state(9)
    shear_degrees = np.degrees(np.arctan(0.2))
    aug = iaa.SomeOf((3, None), [
        iaa.Fliplr(0.5, random_state=seeds[0]),
        iaa.Flipud(0.5, random_state=seeds[1]),
        iaa.Affine(rotate=(-20, 20), random_state=seeds[2], order=0),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=seeds[3], order=0),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=seeds[4], order=0),
        iaa.Affine(shear=(-shear_degrees, shear_degrees), random_state=seeds[5], order=0),
        iaa.Affine(scale=(0.9, 1.1), random_state=seeds[6], order=0),
        iaa.ElasticTransformation(alpha=720, sigma=24, random_state=seeds[7], order=0)
    ], random_order=True, random_state=seeds[8])

    aug = aug.to_deterministic()
    return aug


def sup_aug(in_image: np.ndarray, in_seg_map: np.ndarray, seed: int = None) -> tuple:
    """
    Weak augmentation of the labeled samples, i.e., 2D MRI-images and 2D segmentation maps.
    :param in_image: 2D MRI-images
    :param in_seg_map: 2D segmentation maps
    :param seed: Seed for the augmentation
    :return: Augmented MRI-images and segmentation maps
    """
    image = np.copy(in_image)
    seg_map = np.copy(in_seg_map.astype(np.int32))

    sq = np.random.SeedSequence(seed)

    img_aug = _get_weak_image_augmenter(sq)
    seg_mask_aug = _get_weak_seg_map_augmenter(sq)

    image = img_aug.augment(image=image)
    seg_map = seg_mask_aug(image=seg_map)

    return image, seg_map


def unsup_img_aug(in_image: np.ndarray) -> tuple:
    """
    Strong and weak augmentation of the unlabeled samples, i.e., 2D MRI-images.
    One instance of the MRI-images are weakly augmented and another instance is
    strongly augmented.
    :param in_image: 2D MRI-images
    :return: Weakly and strongly augmented MRI-images and a seed for the strong
    augmentation
    """
    image = np.copy(in_image)

    light_aug = _get_weak_image_augmenter()
    ori_image = light_aug.augment(image=image)

    sq = np.random.SeedSequence()
    aug = _get_strong_image_augmenter(sq)
    aug_image = aug.augment(image=ori_image)

    seed = np.array([sq.entropy], dtype=np.str)
    return ori_image, aug_image, seed


def unsup_logits_aug(in_logits: np.ndarray, str_seeds: list) -> np.ndarray:
    """
    Strong augmentation of predicted logits.
    :param in_logits: Predicted logits
    :param str_seeds: Seeds for the augmentation. In string format because
    limitations in the integer format in Tensorflow.
    :return: Augmented logits
    """
    out_logits = np.zeros(in_logits.shape, np.float32)
    for i in range(in_logits.shape[0]):
        seed = int(str_seeds[i])

        logits = np.copy(in_logits[i, ...])
        aug = _get_strong_seg_map_augmenter(np.random.SeedSequence(seed))
        logits_aug = aug.augment(image=logits)

        # New pixels created in the augmentation should be classified as background
        logits_aug[np.all(logits_aug == [0]*4, axis=-1)] = [100, -100, -100, -100]

        out_logits[i, ...] = logits_aug

    return out_logits


def data_aug(in_data: np.ndarray, str_seeds: list, is_seg_maps: bool) -> np.ndarray:
    """
    Augmentations for MRI-images or segmentation maps, used in the calculation of the
    equivariance Dice score.
    :param in_data: Either MRI-images or segmentation maps
    :param str_seeds: Seeds for the augmentation.
    :param is_seg_maps: True if data is a segmentation map, else False
    :return: Augmented data
    """
    out_data = np.zeros(in_data.shape, np.float32)
    for i in range(in_data.shape[0]):
        seed = int(str_seeds[i])

        data = np.copy(in_data[i, ...])
        if is_seg_maps:
            aug = _get_strong_seg_map_augmenter(np.random.SeedSequence(seed))
        else:
            aug = _get_strong_image_augmenter(np.random.SeedSequence(seed))
        data_aug = aug.augment(image=data)

        out_data[i, ...] = data_aug

    return out_data
