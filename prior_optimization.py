from biggan import model as biggan
from inceptionv1 import model as inceptionv1
import torch
import torchvision
from pytorch_pretrained_biggan import (one_hot_from_names, truncated_noise_sample,
                                       convert_to_images)
import captum.optim as optimviz
import PIL
import numpy as np
import random

# Fix the random seed
torch.manual_seed(1337)
np.random.seed(1337)
random.seed(1337)

# A very basic class that just wraps a single noise vector


class BadInputParametizer(torch.nn.Module):
    def __init__(self):
        super(BadInputParametizer, self).__init__()

        # (1, 256) is shape of BigGAN input for a single batch
        # 128 for noise vector, 128 for class embedding (but we just treat it
        # as an opaque blob here)
        noise = truncated_noise_sample(batch_size=1, dim_z=256)

        noise_t = torch.tensor(noise, requires_grad=True).to("cuda")

        self.v = torch.nn.Parameter(noise_t)

    def forward(self):
        return self.v

# Literally just a stack of alexnet and biggan on top of each other


class GANStack(torch.nn.Module):
    def __init__(self, gan, model):
        super(GANStack, self).__init__()
        self.gan = gan
        self.resize = torchvision.transforms.Resize((224, 224))
        self.regularization = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(4),
            optimviz.transforms.RandomSpatialJitter(8),
            optimviz.transforms.RandomScale(
                scale=(1, 0.975, 1.025, 0.95, 1.05)),
            torchvision.transforms.RandomRotation(degrees=(-5, 5)),
            optimviz.transforms.RandomSpatialJitter(2),
        )
        self.model = model

    def forward(self, x):
        truncation = 0.4
        x = self.gan.generator(x, truncation=truncation)
        x = self.resize(x)
        x = self.regularization(x)
        x = self.model(x)

        return x


stack = GANStack(biggan, inceptionv1)


def max_loss_summarize(loss_value: torch.Tensor):
    return -1 * loss_value.max()


def create_optimized_image(target: torch.nn.Module, channel: int, n_steps: int, lr: float = 0.025) -> torch.Tensor:
    """
    :returns: (1, 256) input ready for biggan insertion, loss_history
    """
    input = BadInputParametizer()
    loss_fn = optimviz.loss.ChannelActivation(target, channel)
    io = optimviz.InputOptimization(
        stack, loss_fn, input, torch.nn.Identity())
    print("Running optim with lr =", lr)
    history = io.optimize(optimviz.optimization.n_steps(
        n_steps, True), lr=lr)  # , loss_summarize_fn=max_loss_summarize)

    return input.v, history


def display_optimized_image(vec: torch.Tensor):
    display(input_to_img(vec))


def input_to_img(vec: torch.Tensor) -> PIL.Image:
    truncation = 0.4
    # Generate an image
    with torch.no_grad():
        output = stack.gan.generator(vec, truncation=truncation)

    # If you have a GPU put back on CPU
    output = output.to('cpu')

    imgs = convert_to_images(output)
    return imgs[0]
