from fastai import *
from fastai.core import *
from fastai.torch_core import *
from fastai.callbacks import hook_outputs
import torchvision.models as models
import matplotlib.pyplot as plt
import torchvision.transforms


class FeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[20, 70, 10]):
        super().__init__()

        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
        requires_grad(self.m_feat, False)
        blocks = [
            i - 1
            for i, o in enumerate(children(self.m_feat))
            if isinstance(o, nn.MaxPool2d)
        ]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel'] + [f'feat_{i}' for i in range(len(layer_ids))]
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()
        

# for Mode Seeking GAN, assume there are pair of images at each input and add regularization term for how similar they are
class FeatureLossMS(nn.Module):

    def __init__(self, z_distance, lam:float=1.0, layer_wgts=[20, 70, 10]):
        super().__init__()
        
        self.z_distance = z_distance
        self.lam = lam
        
        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
        requires_grad(self.m_feat, False)
        blocks = [
            i - 1
            for i, o in enumerate(children(self.m_feat))
            if isinstance(o, nn.MaxPool2d)
        ]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel'] + [f'feat_{i}' for i in range(len(layer_ids))]
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target, z_conc):
        
        split_idx = int(input.size(2) // 2)
        
        input_image_1, input_image_2 = torch.split(input, split_idx, dim=2)
        
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]
        
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        
        original_loss = sum(self.feat_losses)
        lz_nom = torch.mean(torch.abs(input_image_1 - input_image_2))
        lz_denom = self.z_distance
        lz = lz_nom/lz_denom 
        
        eps = 1 * 1e-5 # preventing division by 0
        reg_term = self.lam * (1 / (lz + eps))
        
        return original_loss + reg_term
    
    def update_z(self, z_distance, value):
        setattr(self, z_distance, value)
    
    def setZ(self, z_distance):
        self.z_distance = z_distance

    def __del__(self):
        self.hooks.remove()

