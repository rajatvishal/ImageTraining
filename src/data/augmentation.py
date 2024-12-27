from detectron2.data import transforms as T

def get_augmentations():
    return T.AugmentationList([
        T.Resize((800, 800)),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False),
        T.RandomBrightness(0.8, 1.2),
    ])