import torchvision
import torch
from captum.attr import LayerActivation
from inceptionv1 import model
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from PIL import Image
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"

input_resolution = 224

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=Image.BICUBIC),
        CenterCrop(n_px),
        lambda image: image.convert("RGB"),
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

def activations(layer, img):
    transform = _transform(input_resolution)
    image_input = transform(img).unsqueeze(0).to(device)
    layer_act = LayerActivation(model.forward, layer)
    return layer_act.attribute(image_input)

def act_for_unit(layer, channel_num, image):
    acts = activations(layer, image)[0][channel_num]
    mean = torch.mean(acts).item()
    return mean

acts = {}
layer = model.inception5b
num_channels = 50
num_dataset_examples = 100
for i in range(num_channels):
    for j in range(num_dataset_examples):
        path = "../../../../datasets/inception-featurevis/unit_%s_IMG_%s.png" % (i, j)
        img = Image.open(path)
        act = act_for_unit(layer, i, img)
        key = "Example %s of unit %s" % (j,i)
        acts.update({key:act})


with open('dataset_eg_acts.csv', 'w') as f:
    for key in acts.keys():
        f.write("%s, %s\n" % (key, acts[key]))
        
