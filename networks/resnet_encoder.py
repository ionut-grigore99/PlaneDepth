from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.utils.model_zoo as model_zoo
from pytorch_model_summary import summary
from config.conf import LocalConf


class ResnetEncoder(nn.Module): #this is basically the encoder from the Depth Network
    """
        Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained):
        super(ResnetEncoder, self).__init__()

        # Paper: "we simply use the first five blocks of ResNet50 as the encoder"
        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {
                   18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152
                  }

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4
            
        # self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                       std=[0.229, 0.224, 0.225])

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225 # de unde o luat valorile astea pentru normalizare??
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

if __name__ =='__main__':
    conf = LocalConf().conf
    get = lambda x: conf.get(x)

    model=ResnetEncoder(num_layers=get('model').get('num_resnet_layers'), pretrained=True)
    architecture = summary(model, torch.rand(1, 3, *get('im_sz')), max_depth=4, show_parent_layers=True,
                           print_summary=True)

    onnx=False

    if onnx:
        #Convert to onnx in order to visualize in Netron
        # Input to the model
        x = torch.randn(1, 3, *get('im_sz'), requires_grad=True)
        torch_out = model(x)

        # Export the model
        torch.onnx.export(model,  # model being run
                          x,  # model input (or a tuple for multiple inputs)
                          "resnet_encoder.onnx",  # where to save the model (can be a file or file-like object)
                          export_params=True,  # store the trained parameter weights inside the model file
                          opset_version=10,  # the ONNX version to export the model to
                          do_constant_folding=True,  # whether to execute constant folding for optimization
                          input_names=['input'],  # the model's input names
                          output_names=['output'],  # the model's output names
                          dynamic_axes={'input': {0: 'batch_size'},  # variable length axes
                                        'output': {0: 'batch_size'}})