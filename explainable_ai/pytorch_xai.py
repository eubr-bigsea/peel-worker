import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models

from captum.attr import IntegratedGradients, Occlusion, LayerGradCam, LayerAttribution
from captum.attr import visualization as viz
from matplotlib.colors import LinearSegmentedColormap


class ImageXAI:

    def __init__(self, image, model, labels, transformation=None):
        self.image = image
        self.model = model
        self.labels = labels
        self.transformation = transformation
        self.attributions_ig = None
        self.integrated_gradients = None
        self.attributions_occ = None
        self.upsamp_attr_lgc = None
        self.default_cmap = LinearSegmentedColormap.from_list('custom blue',
                                                              [(0, '#ffffff'),
                                                               (0.25, '#0000ff'),
                                                               (1, '#0000ff')], N=256)

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, image):
        if isinstance(image, str):
            if os.path.isfile(image):
                self._image = Image.open(image)
            else:
                ValueError(f"Class {self.__class__.__name__} does not handle with this {image} as image")
        elif isinstance(image, dict):
            path = image.get("image_path")
            trans_func = image.get("transformation")
            if callable(trans_func) and os.path.isfile(path):
                self._image = trans_func(Image.open(path))
            else:
                ValueError(f"Class {self.__class__.__name__} does not know how to handle with image")
        else:
            raise ValueError(f"Class {self.__class__.__name__} does not know how to handle with image")

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, labels):

        if os.path.exists(labels):
            with open(labels) as json_data:
                self._labels = json.load(json_data)
        else:
            raise ValueError(f"Class {self.__class__.__name__} does not handle with this {labels} as labels")

    def set_all_attributes(self):

        if self.transformation is not None:
            input_img = self.transformation(self.image).unsqueeze(0)
        else:
            input_img = self.image

        self.integrated_gradients = IntegratedGradients(self.model)
        output = self.model(input_img)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        pred_label_idx.squeeze_()
        self.attributions_ig = self.integrated_gradients.attribute(input_img, target=pred_label_idx, n_steps=200)
        occlusion = Occlusion(self.model)
        self.attributions_occ = occlusion.attribute(input_img,
                                                    target=pred_label_idx,
                                                    strides=(3, 8, 8),
                                                    sliding_window_shapes=(3, 15, 15),
                                                    baselines=0)
        layer_gradcam = LayerGradCam(self.model, self.model.layer3[1].conv2)
        attributions_lgc = layer_gradcam.attribute(input_img, target=pred_label_idx)
        self.upsamp_attr_lgc = LayerAttribution.interpolate(attributions_lgc, input_img.shape[2:])

    def viz_grad_img(self):
        _ = viz.visualize_image_attr(np.transpose(self.attributions_ig.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                     np.transpose(self.image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
                                     method='heat_map',
                                     cmap=self.default_cmap,
                                     show_colorbar=True,
                                     sign='positive',
                                     title='Integrated Gradients')
        plt.savefig("teste_pytorch_new.png")
        plt.close()

    def viz_occ(self):

        _ = viz.visualize_image_attr_multiple(
            np.transpose(self.attributions_occ.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            np.transpose(self.image.squeeze().cpu().detach().numpy(), (1, 2, 0)),
            ["original_image", "heat_map", "heat_map", "masked_image"],
            ["all", "positive", "negative", "positive"],
            show_colorbar=True,
            titles=["Original", "Positive Attribution", "Negative Attribution", "Masked"],
            fig_size=(18, 6)
        )

        plt.savefig("teste_pytorch_occ.png")
        plt.close()

    def viz_layer(self):
        _ = viz.visualize_image_attr_multiple(self.upsamp_attr_lgc[0].cpu().permute(1, 2, 0).detach().numpy(),
                                              self.image.permute(1, 2, 0).numpy(),
                                              ["original_image", "blended_heat_map", "masked_image"],
                                              ["all", "positive", "positive"],
                                              show_colorbar=True,
                                              titles=["Original", "Positive Attribution", "Masked"],
                                              fig_size=(18, 6))
        plt.savefig("teste_pytorch_layer.png")
        plt.close()