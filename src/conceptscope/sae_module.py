from src.utils.sae_loader import SAELoader


class SAEModule:
    def __init__(
        self,
        save_root,
        sae_path,
        device,
        backbone,
        dir_name="dataset_analysis",
        checkpoint_dir_name="checkpoints",
        data_root="./data",
    ):
        self.checkpoint_name = sae_path.split("/")[-2]
        self.save_root = save_root
        self.dir_name = dir_name
        self.checkpoint_dir_name = checkpoint_dir_name
        self.checkpoint_path = f"{self.save_root}/{checkpoint_dir_name}/{self.checkpoint_name}"
        self.sae_path = sae_path
        self.device = device
        self.sae, self.vit, self.cfg = SAELoader.get_sae_vit_cfg(sae_path, device, backbone)
        self.data_root = data_root
        self.label_key = "label"
        self.image_key = "image"
