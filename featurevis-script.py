import torch
import torchvision
import captum.optim as optimviz
from inceptionv1 import model
import pickle


def vis_neuron(
    model: torch.nn.Module, target: torch.nn.Module, channel: int
) -> torch.Tensor:
    image = optimviz.images.NaturalImage((224, 224)).to("cuda")
    loss_fn = optimviz.loss.NeuronActivation(target, channel)
    transforms = torch.nn.Sequential(
        torch.nn.ReflectionPad2d(4),
        optimviz.transforms.RandomSpatialJitter(8),
        optimviz.transforms.RandomScale(scale=(1, 0.975, 1.025, 0.95, 1.05)),
        torchvision.transforms.RandomRotation(degrees=(-5, 5)),
        optimviz.transforms.RandomSpatialJitter(2),
        optimviz.transforms.CenterCrop((120, 120)),
    )
    obj = optimviz.InputOptimization(
        model, loss_fn, image, transforms)
    history = obj.optimize(optimviz.optimization.n_steps(128, True))
    return image, history


histories = []

# 5, 20 not optimizable for some reason; left at worse version
for i in range(0, 50):
    # Loop until the optimizer actually makes progress
    while True:
        print("Optimizing", i)
        image, history = vis_neuron(
            model, model.inception5b.branch4[1].conv, i)
        if history[-1] < -1:
            break
    torchvision.transforms.ToPILImage()(image().squeeze()).save(
        f"out/featurevis-chan-{str(i)}.png")
    histories.append(history.cpu().detach().numpy())


with open("featurevis-loss-histories.pickle", "wb") as f:
    pickle.dump(histories, f)
