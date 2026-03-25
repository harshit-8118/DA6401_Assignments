import torch  
import torchvision.models as models 

vgg_model = models.vgg11()

with open('vgg.txt', 'w') as f: 
    f.write(vgg_model.__str__())