import yaml
from segmentator import Segmentator
import torch

# model_path = "/home/ankit/Work/sloam_utils/trained_model/"
# arch_configs = yaml.safe_load(open(model_path+"arch_cfg.yaml", 'r'))
# data_configs = yaml.safe_load(open(model_path+"data_cfg.yaml", 'r'))
# n_classes = len(data_configs["learning_map_inv"])


class Load_Model():
    def __init__(self, model_path):
        # Change model_path variable to point to the trained_model
        self.model_path = model_path
        self.arch_configs = yaml.safe_load(
            open(self.model_path+"arch_cfg.yaml", 'r'))
        self.data_configs = yaml.safe_load(
            open(self.model_path+"data_cfg.yaml", 'r'))
        self.n_classes = len(self.data_configs["learning_map_inv"])

    def load_model(self):
        with torch.no_grad():
            model = Segmentator(ARCH=self.arch_configs,
                            nclasses=self.n_classes, path=self.model_path)

        return model
