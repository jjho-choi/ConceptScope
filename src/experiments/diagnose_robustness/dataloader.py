from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.utils.image_dataset_loader import ImageDatasetLoader


def transform_fn(examples, transform, attribute):
    for key in ["image", "jpg", "webp"]:
        if key in examples:
            images = []
            for img in examples[key]:
                if isinstance(img, str):
                    img = Image.open(img)
                images.append(transform(img.convert("RGB")))
            break
    if attribute is not None:
        labels = examples[attribute]
    else:
        for key in ["label", "cls"]:
            if key in examples:
                labels = examples[key]
                break
    return {"pixel_values": images, "labels": labels}


def get_transforms(image_size, dataset_name=None):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    eval_transforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ]
    )

    return train_transforms, eval_transforms


def get_test_dataset(dataset_name):

    if dataset_name == "imagenet":
        test_dataset = ImageDatasetLoader.load_dataset(dataset_name, seed=1, split="val")
    else:
        test_dataset = ImageDatasetLoader.load_dataset(dataset_name, seed=1, split="test")
    return test_dataset


def get_val_dataset(dataset_name):
    if dataset_name == "celeba":
        test_dataset = ImageDatasetLoader.load_dataset(dataset_name, seed=1, split="valid")
    elif dataset_name == "waterbird":
        test_dataset = ImageDatasetLoader.load_dataset(dataset_name, seed=1, split="val")
    else:
        tot_dataset = ImageDatasetLoader.load_dataset(dataset_name, seed=1)
        find_key = False
        for key in ["val", "valid", "validation"]:
            if key in tot_dataset.keys():
                test_dataset = tot_dataset[key]
                find_key = True
                break
        if not find_key:
            test_dataset = tot_dataset["test"]
    return test_dataset


def get_dataloader(dataset, hf_transform_fn, transforms, attribute, batch_size, num_workers, shuffle):
    dataset = dataset.with_transform(lambda examples: hf_transform_fn(examples, transforms, attribute))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader


def create_dataloaders(
    dataset_name="celeba",
    attribute=None,
    batch_size=32,
    image_size=224,
    num_workers=8,
    test_only=False,
):
    train_transforms, eval_transforms = get_transforms(image_size, dataset_name)

    out = {}
    test_dataset = get_test_dataset(dataset_name)
    test_dataloader = get_dataloader(
        test_dataset, transform_fn, eval_transforms, attribute, batch_size, num_workers, False
    )
    out["test"] = test_dataloader

    if test_only:
        return out

    val_dataset = get_val_dataset(dataset_name)
    val_dataloader = get_dataloader(
        val_dataset, transform_fn, eval_transforms, attribute, batch_size, num_workers, False
    )
    out["val"] = val_dataloader
    train_dataset = ImageDatasetLoader.load_dataset(dataset_name, seed=1, split="train")
    train_dataloader = get_dataloader(
        train_dataset, transform_fn, train_transforms, attribute, batch_size, num_workers, shuffle=True
    )
    out["train"] = train_dataloader
    return out
