import torch
import torch.nn as nn
import timm

import torchvision.transforms as TVF

class MLP(nn.Module):
    def __init__(self, nin, nhidden, nout):
        super().__init__()
        if (nhidden == 0):
            self.main = nn.Linear(nin, nout)
        else:
            self.main = nn.Sequential(
                nn.Linear(nin, nhidden),
                nn.BatchNorm1d(nhidden),
                nn.ReLU(),
                nn.Linear(nhidden, nout)
            )

    def forward(self, x):
        # Our model now returns logits!
        logits = self.main(x)
        return logits
    


class SimpleConvModel(nn.Module):
    def __init__(
        self,
        chin,
        channels,
        num_hidden,
        num_classes
    ):
        super().__init__()
        self.num_classes = num_classes

        def conv(chin, chout, k, s, p):
            return nn.Sequential(
                nn.Conv2d(chin, chout, kernel_size=k, stride=s, padding=p),
                nn.BatchNorm2d(chout),
                nn.ReLU()
            )
        
        self.feature_extractor = nn.Sequential(
            conv(chin, channels, 5, 2, 2),              # 224 -> 112
            conv(channels, channels*2, 3, 2, 1),        # 112 -> 56
            conv(channels*2, channels*4, 3, 2, 1),      # 56 -> 28
            conv(channels*4, channels*8, 3, 2, 1),      # 28 -> 14
            conv(channels*8, channels*16, 3, 2, 1),     # 14 -> 7
        )

        self.num_features = channels*16

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),                # 7x7 -> 1x1
            nn.Flatten(),                               # <B,C,1,1> -> <B,C>
            MLP(
                nin=1*1*self.num_features,
                nhidden=num_hidden,
                nout=num_classes
            )
        )


    def forward(self, x):
        f = self.feature_extractor(x)
        logits = self.head(f)
        return logits
    

class PretrainedConvModel(nn.Module):
    def __init__(
        self,
        num_hidden,
        num_classes
    ):
        super().__init__()
        self.num_classes = num_classes

        # Create our pre-trained model
        self.feature_extractor = timm.create_model("resnet18", pretrained=True)
        self.feature_extractor.reset_classifier(0, "")

        # Freeze parameters
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        self.norm_transform = TVF.Normalize(
            mean=self.feature_extractor.default_cfg["mean"],
            std=self.feature_extractor.default_cfg["std"]
        )

        # Replace our own classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),                # 7x7 -> 1x1
            nn.Flatten(),                               # <B,C,1,1> -> <B,C>
            MLP(
                nin=1*1*self.feature_extractor.num_features,
                nhidden=num_hidden,
                nout=num_classes
            )
        )


    def forward(self, x):
        x = self.norm_transform(x)
        f = self.feature_extractor(x)
        logits = self.head(f)
        return logits
    


class VitModel(nn.Module):
    def __init__(
        self,
        num_classes
    ):
        super().__init__()
        self.num_classes = num_classes

        # Create our pre-trained model
        self.feature_extractor = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k", pretrained=True)
        self.feature_extractor.reset_classifier(0, "")

        self.norm_transform = TVF.Normalize(
            mean=self.feature_extractor.default_cfg["mean"],
            std=self.feature_extractor.default_cfg["std"]
        )

        # Freeze parameters
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

        # Replace our own classification head
        self.head = nn.Sequential(
            MLP(
                nin=self.feature_extractor.num_features,
                nhidden=0,
                nout=num_classes
            )
        )


    def forward(self, x):
        x = self.norm_transform(x)
        f = self.feature_extractor(x)
        f = f[:,0]      # We take only class token
        logits = self.head(f)
        return logits

