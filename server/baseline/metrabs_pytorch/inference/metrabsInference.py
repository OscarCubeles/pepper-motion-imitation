
import numpy as np
import pickle
import torch

import metrabs_pytorch.backbones.efficientnet as effnet_pt
import metrabs_pytorch.models.metrabs as metrabs_pt
from metrabs_pytorch.multiperson import multiperson_model
from metrabs_pytorch.util import get_config
from metrabs_pytorch.joint_info import  JointInfo


class metrabs_inference(object):

    def __init__(self,model_dir):
        self.model_dir = model_dir
        get_config(model_dir+'/config.yaml')
        self.multiperson_model_pt = self.load_multiperson_model().cuda()


    def load_model(self):
        return  self.multiperson_model_pt


    def load_multiperson_model(self):
        model_pytorch = self.load_crop_model()
        skeleton_infos = pickle.load(open(self.model_dir+'/skeleton_infos.pkl', "rb"))
        joint_transform_matrix = np.load(self.model_dir+'/joint_transform_matrix.npy')

        with torch.device('cuda'):
            return multiperson_model.Pose3dEstimator(
                model_pytorch.cuda(), skeleton_infos, joint_transform_matrix)


    def load_crop_model(self):
        ji_np = np.load(self.model_dir+'/joint_info.npz')
        ji = JointInfo(ji_np['joint_names'], ji_np['joint_edges'])
        backbone_raw = getattr(effnet_pt, f'efficientnet_v2_l')()
        preproc_layer = effnet_pt.PreprocLayer()
        backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
        model = metrabs_pt.Metrabs(backbone, ji)
        model.eval()
        model.load_state_dict(torch.load(self.model_dir+'/ckpt.pt', weights_only=True))
        return model


# if __name__ == '__main__':
#     model_dir = r"..\metrabs\metrabs_eff2l_384px_800k_28ds_pytorch"
#     inference = metrabs_inference(model_dir)
#     model = inference.load_model()
#     print(model)