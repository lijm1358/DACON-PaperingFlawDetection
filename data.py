import albumentations as A
import cv2
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import Dataset


class TrainAugmentation:
    def __init__(self, resize):
        print(resize)
        self.transform = A.Compose(
            [
                A.Resize(resize[0], resize[1]),
                A.Normalize(
                    mean=(0.601, 0.593, 0.577),
                    std=(0.087, 0.091, 0.095),
                    max_pixel_value=255.0,
                    always_apply=False,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        return self.transform(image=image)


class TestAugmentation:
    def __init__(self, resize):
        self.transform = A.Compose(
            [
                A.Resize(resize[0], resize[1]),
                A.Normalize(
                    mean=(0.601, 0.593, 0.577),
                    std=(0.087, 0.091, 0.095),
                    max_pixel_value=255.0,
                    always_apply=False,
                    p=1.0,
                ),
                ToTensorV2(),
            ]
        )

    def __call__(self, image):
        return self.transform(image=image)


class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        img_path = self.img_path_list[index]

        image = cv2.imread(img_path)

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image

    def __len__(self):
        return len(self.img_path_list)
