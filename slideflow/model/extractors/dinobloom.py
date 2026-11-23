import timm
import torch

from slideflow.model.extractors._factory_torch import TorchFeatureExtractor


# -----------------------------------------------------------------------------

class DinoBloomFeatures(TorchFeatureExtractor):
    """DinoBloom pretrained feature extractor in 4 sizes (small, base, large, giant).

    This class is used to extract tile-level features from DinoBloom.

    Feature dimensions: 384, 768, 1024, 1536

    Manuscript: https://arxiv.org/abs/2404.05022

    Github: https://github.com/marrlab/DinoBloom
    """

    tag = 'dinobloom'
    license = """Apache License 2.0 (License available at https://github.com/marrlab/DinoBloom)"""
    citation = """
@misc{dinobloom,
      title={DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology},
      author={Valentin Koch and Sophia J. Wagner and Salome Kazeminia and Ece Sancar and Matthias Hehr and Julia Schnabel and Tingying Peng and Carsten Marr},
      year={2024},
      eprint={2404.05022},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
"""
    _feature_sizes = {
        "small": 384,
        "base": 768,
        "large": 1024,
        "giant": 1536,
    }

    def __init__(self, model_size: str = 'small', weights=None, pretrained: bool = True, device='cuda', **kwargs):
        super().__init__(**kwargs)

        if model_size not in self._feature_sizes:
            raise ValueError(
                f"Invalid DinoBloom model_size '{model_size}'. "
                f"Expected one of {list(self._feature_sizes.keys())}."
            )

        from slideflow.model import torch_utils

        self.device = torch_utils.get_device(device)
        self._model_size = model_size
        self._weights = weights

        if weights is None and pretrained:
            self.model = timm.create_model(
                f"hf-hub:1aurent/vit_{model_size}_patch14_224.dinobloom",
                pretrained=True,
            )
        else:
            self.model = timm.create_model(
                f"vit_{model_size}_patch14_dinov2",
                pretrained=False,
                img_size=224,
                num_classes=0,
            )
            if weights is not None:
                state_dict = torch.load(weights, map_location=self.device)
                self.model.load_state_dict(state_dict, strict=True)

        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------------------
        self.num_features = int(self._feature_sizes[model_size])
        self.transform = self.build_transform(
            img_size=224,
            center_crop=True,
            interpolation='bicubic',
            norm_mean=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
        )
        self.preprocess_kwargs = dict(standardize=False)

    def dump_config(self):
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``slideflow.build_feature_extractor()``.
        """
        return self._dump_config(
            class_name='slideflow.model.extractors.dinobloom.DinoBloomFeatures',
            model_size=self._model_size,
            weights=self._weights,
        )
