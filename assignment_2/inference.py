"""Inference and evaluation
"""
import torch 

model_path = 'checkpoints/classifier.pth'
ckpt = torch.load(model_path, map_location='cpu', weights_only=True)
state_dict = ckpt['state_dict']
for key in state_dict.keys():
    print(key, state_dict[key].shape)