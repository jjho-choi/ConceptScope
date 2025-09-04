import os
from glob import glob

import pandas as pd
from datasets import ClassLabel, Dataset, Features, Value, load_dataset, load_from_disk
from tqdm import tqdm


class ImageDatasetLoader:

    @staticmethod
    def _load_huggingface_local_dataset(dataset_name: str, seed: int, split: str = "train"):
        dataset = load_from_disk(dataset_name)[split]
        dataset = dataset.shuffle(seed=seed)

        return dataset

    @staticmethod
    def _load_things_dataset(data_root):
        class_names = sorted(os.listdir(data_root))
        label2id = {name: idx for idx, name in enumerate(class_names)}
        data_dict = {"image": [], "label": []}
        for class_name in class_names:
            image_files = glob(os.path.join(data_root, class_name, "*.jpg"))
            for image_file in image_files:
                data_dict["image"].append(image_file)
                data_dict["label"].append(label2id[class_name])
        features = Features(
            {
                "image": Value("string"),
                "label": ClassLabel(names=class_names),
            }
        )
        dataset = Dataset.from_dict(data_dict, features=features)
        return dataset

    @staticmethod
    def _load_stanford_action(data_root, split):
        with open(f"{data_root}/ImageSplits/{split}.txt", "r") as f:
            file_list = f.readlines()
        label_names = sorted(list(set([fname.rsplit("_", 1)[0] for fname in file_list])))
        label2id = {name: idx for idx, name in enumerate(label_names)}
        data_dict = {"image": [], "label": []}
        for file_name in file_list:
            file_name = file_name.strip()
            label = label2id[file_name.rsplit("_", 1)[0]]
            image_path = os.path.join(data_root, "JPEGImages", file_name)
            data_dict["image"].append(image_path)
            data_dict["label"].append(label)
        features = Features(
            {
                "image": Value("string"),
                "label": ClassLabel(names=label_names),
            }
        )
        dataset = Dataset.from_dict(data_dict, features=features)
        return dataset

    @staticmethod
    def _load_raf_dataset(data_root, split):
        data_dict = {"image": [], "label": []}
        file_list = glob(f"{data_root}/{split}/*/*")
        label_names = ["surprise", "fear", "disgust", "happy", "sad", "angry", "neutral"]
        for image_path in file_list:
            label = int(image_path.split("/")[-2]) - 1
            data_dict["label"].append(label)
            data_dict["image"].append(image_path)
        features = Features(
            {
                "image": Value("string"),
                "label": ClassLabel(names=label_names),
            }
        )
        return Dataset.from_dict(data_dict, features=features)

    @staticmethod
    def _load_waterbird_dataset(data_root: str, split: str = "train", target_attribute: str = "species"):
        if target_attribute is None:
            target_attribute = "species"
        split_mapping = {"train": 0, "val": 1, "test": 2}
        if split not in split_mapping:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")

        metadata = pd.read_csv(os.path.join(data_root, "metadata.csv"))
        metadata = metadata[metadata["split"] == split_mapping[split]]

        bbox = pd.read_csv(os.path.join(data_root, "bounding_boxes.txt"), sep=" ", header=None)
        bbox.columns = ["image_id", "x", "y", "width", "height"]

        place_names = sorted(list(set(metadata["place_filename"].apply(lambda x: x.split("/")[2]))))
        place2id = {name: idx for idx, name in enumerate(place_names)}
        data_dict = {"image": [], "species": [], "place": [], "detailed_place": []}

        for i, row in tqdm(metadata.iterrows(), desc="Loading Waterbird dataset"):
            try:
                image = os.path.join(data_root, row["img_filename"])
                data_dict["image"].append(image)
                data_dict["species"].append(row["y"])
                data_dict["place"].append(row["place"])
                detailed_place = place2id[row["place_filename"].split("/")[2]]
                data_dict["detailed_place"].append(detailed_place)

            except Exception as e:
                print(f"Error loading image {row['img_filename']}: {e}")

        if target_attribute == "species":
            data_dict["label"] = data_dict.pop("species")
            dataset = Dataset.from_dict(data_dict)
        else:
            data_dict["label"] = data_dict.pop("detailed_place")
            place_names = [" ".join(name.split("_")).lower() for name in place_names]
            features = Features(
                {
                    "image": Value("string"),
                    "label": ClassLabel(names=place_names),
                    "place": Value("string"),
                    "species": Value("int32"),
                }
            )
            dataset = Dataset.from_dict(data_dict, features=features)

        return dataset

    @staticmethod
    def _load_celeba(dataset_name: str, split: str, target_attribute: str = "Blond_Hair", seed: int = 1):
        dataset = ImageDatasetLoader._load_huggingface_dataset(dataset_name, seed, split=split)
        if target_attribute is None:
            target_attribute = "Blond_Hair"
        for feature in dataset.features:
            if feature == target_attribute:
                dataset = dataset.rename_column(feature, "label")
        return dataset

    @staticmethod
    def _load_huggingface_dataset(dataset_name: str, seed: int, split: str = "train"):
        dataset = load_dataset(dataset_name, split=split)
        dataset = dataset.shuffle(seed=seed)
        if "jpg" in dataset.features:
            dataset = dataset.rename_column("jpg", "image")
        elif "webp" in dataset.features:
            dataset = dataset.rename_column("webp", "image")
        if "cls" in dataset.features:
            dataset = dataset.rename_column("cls", "label")
        return dataset

    @staticmethod
    def _load_ms_coco(dataset_name: str, seed: int, split: str = "train"):
        dataset = ImageDatasetLoader._load_huggingface_dataset(dataset_name, seed, split=split)
        class_names = ["person", "bird", "dog", "cat", "car", "airplane"]

        import spacy

        nlp = spacy.load("en_core_web_sm")

        def add_binary_labels(example):
            doc = nlp(example["txt"].lower())
            lemmas = {token.lemma_ for token in doc}
            return {class_name: int(class_name in lemmas) for class_name in class_names}

        dataset = dataset.map(add_binary_labels)

        return dataset

    @staticmethod
    def load_dataset(dataset_name: str, seed: int = 1, split: str = None, target_attribute: str = None, root="./data"):
        """Load dataset and get classnames."""
        dataset_loaders = {
            "ms_coco": lambda: ImageDatasetLoader._load_ms_coco(
                "clip-benchmark/wds_mscoco_captions2017", seed, split=split
            ),
            "waterbird": lambda: ImageDatasetLoader._load_waterbird_dataset(
                f"{root}/image_dataset/waterbird", split=split, target_attribute=target_attribute
            ),
            "things": lambda: ImageDatasetLoader._load_things_dataset(
                "/home/nas4_user/jinhochoi/dataset/things/images"
            ),
            "nico_95": lambda: ImageDatasetLoader._load_huggingface_local_dataset(
                f"{root}/image_dataset/nico/super_95", seed, split=split
            ),
            "nico_75": lambda: ImageDatasetLoader._load_huggingface_local_dataset(
                f"{root}/image_dataset/nico/super_75", seed, split=split
            ),
            "nico_90": lambda: ImageDatasetLoader._load_huggingface_local_dataset(
                f"{root}/image_dataset/nico/super_90", seed, split=split
            ),
            "celeba": lambda: ImageDatasetLoader._load_celeba("flwrlabs/celeba", split, target_attribute, seed),
            "imagenet-v2": lambda: ImageDatasetLoader._load_huggingface_dataset(
                "clip-benchmark/wds_imagenetv2", seed, split=split
            ),
            "imagenet-sketch": lambda: ImageDatasetLoader._load_huggingface_dataset(
                "clip-benchmark/wds_imagenet_sketch", seed, split=split
            ),
            "imagenet": lambda: ImageDatasetLoader._load_huggingface_dataset(
                "evanarlian/imagenet_1k_resized_256", seed, split=split
            ),
            "stanford_action": lambda: ImageDatasetLoader._load_stanford_action(
                f"{root}/image_dataset/Stanford40", split
            ),
            "food101": lambda: ImageDatasetLoader._load_huggingface_dataset("zguo0525/food101", seed, split=split),
            "sun397": lambda: ImageDatasetLoader._load_huggingface_dataset("dpdl-benchmark/sun397", seed, split=split),
            "caltech101": lambda: ImageDatasetLoader._load_huggingface_dataset(
                "dpdl-benchmark/caltech101", seed, split=split
            ),
            "dtd": lambda: ImageDatasetLoader._load_huggingface_dataset("tanganke/dtd", seed, split=split),
            "raf-db": lambda: ImageDatasetLoader._load_raf_dataset("data/image_dataset/RAF-DB/DATASET", split=split),
        }

        if dataset_name in dataset_loaders:
            dataset = dataset_loaders[dataset_name]()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        return dataset
