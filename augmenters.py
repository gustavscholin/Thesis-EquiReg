import numpy as np
import imgaug.augmenters as iaa


def _get_light_image_augmenter(sq=np.random.SeedSequence()):
    seeds = sq.generate_state(5)
    aug = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=seeds[0]),
        iaa.Flipud(0.5, random_state=seeds[1]),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=seeds[2]),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=seeds[3])
    ], random_order=True, random_state=seeds[4])

    aug = aug.to_deterministic()
    return aug


def _get_light_seg_mask_augmenter(sq):
    seeds = sq.generate_state(5)
    aug = iaa.Sequential([
        iaa.Fliplr(0.5, random_state=seeds[0]),
        iaa.Flipud(0.5, random_state=seeds[1]),
        iaa.Affine(translate_percent={"x": (-0.1, 0.1)}, random_state=seeds[2], order=0),
        iaa.Affine(translate_percent={"y": (-0.1, 0.1)}, random_state=seeds[3], order=0)
    ], random_order=True, random_state=seeds[4])

    aug = aug.to_deterministic()
    return aug


def _get_image_augmenter(sq):
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


def _get_seg_mask_augmenter(sq):
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


def sup_aug(in_image, in_seg_mask, entropy=None):
    image = np.copy(in_image)
    seg_mask = np.copy(in_seg_mask.astype(np.int32))

    sq = np.random.SeedSequence(entropy)

    img_aug = _get_light_image_augmenter(sq)
    seg_mask_aug = _get_light_seg_mask_augmenter(sq)

    image = img_aug.augment(image=image)
    seg_mask = seg_mask_aug(image=seg_mask)

    return image, seg_mask


def unsup_img_aug(in_image):
    image = np.copy(in_image)

    light_aug = _get_light_image_augmenter()
    ori_image = light_aug.augment(image=image)

    sq = np.random.SeedSequence()
    aug = _get_image_augmenter(sq)
    aug_image = aug.augment(image=ori_image)

    entropy = np.array([sq.entropy], dtype=np.str)
    return ori_image, aug_image, entropy


def seg_aug(in_segs, entropy):
    out_segs = np.zeros(in_segs.shape, np.float32)
    for i in range(in_segs.shape[0]):
        seed_ent = int(entropy[i])

        seg = np.copy(in_segs[i, ...])
        aug = _get_seg_mask_augmenter(np.random.SeedSequence(seed_ent))
        seg_aug = aug.augment(image=seg)

        out_segs[i, ...] = seg_aug

    return out_segs


def img_aug(in_imgs, entropy, is_seg_maps):
    out_imgs = np.zeros(in_imgs.shape, np.float32)
    for i in range(in_imgs.shape[0]):
        seed_ent = int(entropy[i])

        img = np.copy(in_imgs[i, ...])
        if is_seg_maps:
            aug = _get_seg_mask_augmenter(np.random.SeedSequence(seed_ent))
        else:
            aug = _get_image_augmenter(np.random.SeedSequence(seed_ent))
        img_aug = aug.augment(image=img)

        out_imgs[i, ...] = img_aug

    return out_imgs


def unsup_logits_aug(in_logits, entropy):
    out_logits = np.zeros(in_logits.shape, np.float32)
    for i in range(in_logits.shape[0]):
        seed_ent = int(entropy[i])

        logits = np.copy(in_logits[i, ...])
        aug = _get_seg_mask_augmenter(np.random.SeedSequence(seed_ent))
        logits_aug = aug.augment(image=logits)

        logits_aug[np.all(logits_aug == [0]*4, axis=-1)] = [100, -100, -100, -100]
        out_logits[i, ...] = logits_aug

    return out_logits

