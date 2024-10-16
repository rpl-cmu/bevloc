# Third Party
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2
import numpy as np

# In House
from aeromatch.features.encoding import FeatureExtractor, Backbone

class AeroBEVNetBase(nn.Module):
    """
    General skeleton for an AeroBevNet
    """

    def __init__(self):
        super().__init__()
        self.encoder = None
        self.fe = None

        # Preprocessing 
        self.preprocess = v2.Compose([
        # v2.RandomResizedCrop(size=(224, 224), antialias=True),
        # v2.RandomHorizontalFlip(p=0.5),
        # v2.Resize((224,224), antialias=False),
        v2.ToDtype(torch.float32),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def embed_a(self, x):
        """
        Embed via a FC layer plus L2 normalization
        """
        x = self.fc_a(x.view(x.shape[0], -1))
        x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x):
        """
        Forward Pass to receive the encoding
     """
        x = self.preprocess(x).to("cuda")
        # return self.embed_a(self.fe.get_feature_maps(x.unsqueeze(1), (224,224)))
        return self.embed_a(self.fe.get_feature_maps(x.unsqueeze(1)))
    
    def get_model_params(self):
        return self.encoder.parameters()

class AeroBEVNet(AeroBEVNetBase):
    """
    This is the aerial encoder for creating embeddings.
    We look for more fine estimates and use a more geometry perserving encoder.
    """
    def __init__(self, feat_size, embedding_size, grid_size, backbone, device = "cuda"):
        super().__init__()
        backbone = Backbone.eDINO_B

        # Specific to which backbone is used
        if backbone == Backbone.eDINO_B or backbone == Backbone.eDINO_S:
            self.spatial_res = (grid_size[0]//14, grid_size[1]//14)
            if backbone == Backbone.eDINO_B:
                feat_size = 768
            elif backbone == Backbone.eDINO_S:
                feat_size = 384
            elif backbone == Backbone.eDINO_L:
                feat_size=1024
            elif backbone == Backbone.eDINO_G:
                feat_size=1536
        else:
            self.spatial_res = (224,224)
        spatial_sz  = np.prod(np.array(self.spatial_res))

        # Feature encoding
        if feat_size is not None:
            self.fe = FeatureExtractor(feat_size, backbone, device)
            self.encoder = self.fe.encoder

        # Create the embedding FC layer
        self.fc_a = nn.Linear(feat_size*spatial_sz, embedding_size).to(device)

    def forward(self, x):
        return super().forward(x)