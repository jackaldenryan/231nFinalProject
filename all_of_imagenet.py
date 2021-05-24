from turtle import forward
import torchvision
from torchvision import transforms
import torch
from inceptionv1 import model as inceptionv1
import collections

imagenet = torchvision.datasets.ImageNet(
    root="/datasets/imagenet", split="train",
    transform=torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]))
dataloader = torch.utils.data.DataLoader(imagenet, batch_size=64,
                                         shuffle=False)

layers_of_interest = {
    "inception5b": inceptionv1.inception5b,
    "5bconv": inceptionv1.inception5b.branch4[1].conv,
    "inception4b": inceptionv1.inception4b,
    "fc": inceptionv1.fc
}


def layer_obj_to_name(layer: torch.nn.Module):
    for name, layer2 in layers_of_interest.items():
        if layer == layer2:
            return name
    raise ArgumentError()


def forward_hook(self, input, output):
    print(self)


activations = collections.defaultdict(dict)


for idx, batch in enumerate(dataloader):
    images, labels = batch
    image_indexes = idx + torch.range(0, len(images))
    images = images.to("cuda")
    labels = labels.to("cuda")

    def forward_hook(layer, input, output: torch.Tensor):
        layer_name = layer_obj_to_name(layer)
        if len(output.shape) > 2:
            output = output.mean(dim=2).mean(dim=2)
        assert len(output.shape) == 2

        for image_idx in image_indexes:
            activations[layer_name][image_idx] = output.cpu().detach().numpy()

    handles = []
    for layer in layers_of_interest:
        handles.append(
            layers_of_interest[layer].register_forward_hook(forward_hook))

    inceptionv1(images)

    for handle in handles:
        handle.remove()

    if idx % 10 == 0:
        print("Batch", idx, "idx", image_indexes[0])
