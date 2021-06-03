from turtle import forward
import torchvision
from torchvision import transforms
import torch
from inceptionv1 import model as inceptionv1
import collections
import pickle


def dataset_with_indices(cls):
    """
    Modifies the given Dataset class to return a tuple data, target, index
    instead of just data, target.
    """

    def __getitem__(self, index):
        data, target = cls.__getitem__(self, index)
        return data, target, index

    return type("ImageNet2", (cls,), {
        '__getitem__': __getitem__,
    })


ImageNet = dataset_with_indices(torchvision.datasets.ImageNet)
imagenet = ImageNet(
    root="/datasets/imagenet", split="train",
    transform=torchvision.transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ]))
torch.manual_seed(1)
dataloader = torch.utils.data.DataLoader(imagenet, batch_size=64,
                                         shuffle=True)

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


activations = collections.defaultdict(dict)


for idx, batch in enumerate(dataloader):
    images, labels, indexes = batch
    count = idx * len(images)
    if count >= 10_000:
        break
    images = images.to("cuda")
    labels = labels.to("cuda")

    def forward_hook(layer, input, output: torch.Tensor):
        layer_name = layer_obj_to_name(layer)
        if len(output.shape) > 2:
            output = output.mean(dim=2).mean(dim=2)
        assert len(output.shape) == 2

        for image_idx in range(0, len(images)):
            activations[layer_name][int(
                indexes[image_idx])] = output[image_idx].cpu().detach().numpy()

    handles = []
    for layer in layers_of_interest:
        handles.append(
            layers_of_interest[layer].register_forward_hook(forward_hook))

    inceptionv1(images)

    for handle in handles:
        handle.remove()

    if idx % 10 == 0:
        print("Batch", idx, "count", count)


with open("outputs/imagenet-activations.pickle", "wb") as f:
    pickle.dump(activations, f)
    print("Successfully dumped activations")
