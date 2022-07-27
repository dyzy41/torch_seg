import albumentations as albu
from albumentations.pytorch import ToTensorV2

def train_aug(mean, std):
    transform = [
        albu.Flip(p=1),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=1, border_mode=0),
        albu.IAAAdditiveGaussianNoise(p=0.2),
        albu.IAAPerspective(p=0.5),
        albu.RandomGridShuffle(),
        albu.Cutout(max_h_size=32, max_w_size=32),
        albu.OneOf(
            [
                albu.CLAHE(p=1),
                albu.RandomBrightness(p=1),
                albu.RandomGamma(p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.IAASharpen(p=1),
                albu.Blur(blur_limit=3, p=1),
                albu.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),
        albu.OneOf(
            [
                albu.RandomContrast(p=1),
                albu.HueSaturationValue(p=1),
            ],
            p=0.9,
        ),
        albu.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    return albu.Compose(transform)


def val_aug(mean, std):
    transform = [
        albu.Normalize(mean=mean, std=std),
        ToTensorV2()
    ]
    return albu.Compose(transform)
