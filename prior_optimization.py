from alexnet import model as alexnet
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
        v = torch.tensor(truncated_noise_sample(
            truncation=0.4, batch_size=1), requires_grad=True).to("cuda")
        class_vec = biggan.embeddings(torch.tensor(one_hot_from_names(
            ["coffee"], batch_size=1), requires_grad=True).to("cuda"))
        self.v = torch.nn.Parameter(torch.cat((v, class_vec), dim=1))

        print(self.v.shape)

    def forward(self):
        return self.v

# Literally just a stack of alexnet and biggan on top of each other


class GANStack(torch.nn.Module):
    def __init__(self, gan, model):
        super(GANStack, self).__init__()
        self.gan = gan
        self.resize = torchvision.transforms.Resize((224, 224))
        self.model = model

    def forward(self, noise_vector):
        truncation = 0.4
        output = self.gan.generator(noise_vector, truncation=truncation)
        output = self.resize(output)
        output = self.model(output)

        return output


stack = GANStack(biggan, alexnet)


def create_optimized_image(target: torch.nn.Module, channel: int, n_steps: int) -> torch.Tensor:
    """
    :returns: (1, 256) input ready for biggan insertion
    """
    input = BadInputParametizer()
    loss_fn = optimviz.loss.ChannelActivation(target, channel)
    io = optimviz.InputOptimization(
        stack, loss_fn, input, torch.nn.Identity())
    io.optimize(optimviz.optimization.n_steps(128, True))

    return input.v


def display_optimized_image(vec: torch.Tensor):
    truncation = 0.4
    # Generate an image
    with torch.no_grad():
        output = stack.gan.generator(vec, truncation=truncation)

    # If you have a GPU put back on CPU
    output = output.to('cpu')

    imgs = convert_to_images(output)
    display(imgs[0])
