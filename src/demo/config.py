class AppConfig:
    dataset_options = ["ImageNet", "SUN397", "Food101", "Waterbird"]
    default_dataset = "ImageNet"
    mid_font_size = 18


class ClassOverviewConfig:
    height = 450
    min_top_k_dist = 1
    max_top_k_dist = 20
    default_top_k_dist = 10
    step_top_k_dist = 1


class ConceptViewConfig:
    thumbnail_width = 80
    default_top_k_samples = 20
    max_top_k_samples = 20
    min_top_k_samples = 1
    step_top_k_samples = 1
    image_thumbnail_width = 100


class ClassSampleViewConfig:
    min_top_k_samples = 10
    max_top_k_samples = 50
    default_top_k_samples = 20
    step_top_k_samples = 1


class ConceptListViewConfig:
    height = 850
