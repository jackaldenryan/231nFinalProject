from clipv import model as clip
from biggan import model as biggan
import torch
import torchvision
from pytorch_pretrained_biggan import (one_hot_from_names, truncated_noise_sample,
                                       convert_to_images)
import captum.optim as optimviz


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


stack = GANStack(biggan, clip.visual)


def create_optimized_image(target: torch.nn.Module, channel: int, n_steps: int) -> torch.Tensor:
    """
    :returns: (1, 256) input ready for biggan insertion, loss_history
    """
    input = BadInputParametizer()
    loss_fn = optimviz.loss.ChannelActivation(target, channel)
    io = optimviz.InputOptimization(
        stack, loss_fn, input, torch.nn.Identity())
    history = io.optimize(optimviz.optimization.n_steps(n_steps, True))

    return input.v, history


def display_optimized_image(vec: torch.Tensor):
    truncation = 0.4
    # Generate an image
    with torch.no_grad():
        output = stack.gan.generator(vec, truncation=truncation)

    # If you have a GPU put back on CPU
    output = output.to('cpu')

    imgs = convert_to_images(output)
    display(imgs[0])
