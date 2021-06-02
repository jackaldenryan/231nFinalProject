from lib2to3.pytree import convert
from biggan import model as biggan
from inceptionv1 import model as inceptionv1
import torch
import torchvision
from pytorch_pretrained_biggan import (one_hot_from_names, truncated_noise_sample,
                                       convert_to_images)
import captum.optim as optimviz
import captum.attr as attr
import PIL
import numpy as np
from IPython.display import clear_output
from resnet import resnet18

from torchvision import transforms

# Fix the random seed
# torch.manual_seed(1337)
# np.random.seed(1337)
# random.seed(1337)

truncation = 1


# A very basic class that just wraps a single noise vector


# Literally just a stack of alexnet and biggan on top of each other


class GANStack(torch.nn.Module):
    def __init__(self, gan, model):
        super(GANStack, self).__init__()
        self.gan = gan
        self.regularization = torch.nn.Sequential(
            # torch.nn.ReflectionPad2d(4),
            optimviz.transforms.RandomSpatialJitter(8),
            optimviz.transforms.RandomScale(
                scale=(1, 0.975, 1.025, 0.95, 1.05)),
            torchvision.transforms.RandomRotation(degrees=(-5, 5)),
            optimviz.transforms.RandomSpatialJitter(2),
        )
        self.convert = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                0.229, 0.224, 0.225]),
        ])
        self.model = model
        self.steps = 0

    def forward(self, x):
        if self.steps % 100 == 0:
            display_optimized_image(x)
        x = self.gan.generator(x, truncation=truncation)
        # Gan output has 1x1x256x256, with values between -1 and 1
        # We need to transform this with InceptionV1's transforms
        x = ((x + 1.0) / 2.0 * 256)  # Convert to 0-255
        x = self.convert(x)
        x = self.regularization(x)
        x = self.model(x)

        self.steps += 1

        return x


stack = GANStack(biggan, inceptionv1)


class SimpleInputParametizer(torch.nn.Module):
    def __init__(self):
        super(SimpleInputParametizer, self).__init__()

        # (1, 256) is shape of BigGAN input for a single batch
        # 128 for noise vector, 128 for class embedding (but we just treat it
        # as an opaque blob here)
        noise = truncated_noise_sample(batch_size=1, dim_z=128)
        self.class_vector = torch.nn.Parameter(
            torch.zeros((128,), requires_grad=True).to("cuda"))
        self.noise_t = torch.nn.Parameter(
            torch.tensor(noise, requires_grad=True).to("cuda"))

    def forward(self):
        noise_t = self.noise_t.clamp(-truncation, truncation)
        return torch.cat([noise_t.squeeze(), self.class_vector])


class ResNetEmbedding(torch.nn.Module):
    def __init__(self):
        super(ResNetEmbedding, self).__init__()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                                 0.229, 0.224, 0.225]),
        ])

    def forward(self, x):
        x = self.preprocess(x)
        x = resnet18(x)
        return x


class ChannelActivationWithSimilarity(torch.nn.Module):
    def __init__(self,
                 input_parametizer: SimpleInputParametizer,
                 target: torch.nn.Module, channel: int,
                 img: torch.Tensor,
                 anchor: float):
        super(ChannelActivationWithSimilarity, self).__init__()
        self.anchor = anchor
        self.target = target
        self.channel = channel
        self.input = input_parametizer
        self.img = transforms.ToTensor()(img)
        self.img = self.img.reshape(1, *self.img.shape).to("cuda")
        self.loss_fn = optimviz.loss.ChannelActivation(target, channel)
        self.resnet_embedding = ResNetEmbedding()
        self.steps = 0

    def similarity(self):
        gan_img = (stack.gan.generator(
            self.input(), truncation=truncation))
        # Convert to 0-255
        gan_img = ((gan_img + 1.0) / 2.0 * 256)

        x = self.resnet_embedding.forward(gan_img).reshape(-1)
        y = self.resnet_embedding.forward(self.img).reshape(-1)
        return torch.dot(x, y) / (torch.norm(x) * torch.norm(y))

    def forward(self, mappings):
        x = self.loss_fn(mappings)
        # Add similarity score to img
        similarity = self.similarity()
        if self.steps % 10 == 0:
            print(similarity)
        x += self.anchor * similarity
        self.steps += 1
        return x


def create_optimized_image(target: torch.nn.Module, img: torch.Tensor, channel: int, n_steps: int, lr: float = 0.025, anchor: float = 1.0, verbose: bool = True) -> torch.Tensor:
    """
    :returns: (1, 256) input ready for biggan insertion, loss_history
    """
    input = SimpleInputParametizer()

    # Calculating how close the images are
    loss_fn = ChannelActivationWithSimilarity(
        input, target, channel, img, anchor)
    io = optimviz.InputOptimization(
        stack, loss_fn, input, torch.nn.Identity())
    print("Running optim with lr =", lr)
    history = io.optimize(optimviz.optimization.n_steps(
        n_steps, verbose), optimizer=torch.optim.Adam(io.parameters(), lr=lr))  # , loss_summarize_fn=max_loss_summarize)

    return input.forward(), history, input.class_vector


def display_optimized_image(vec: torch.Tensor):
    # clear_output(wait=True)
    display(input_to_img(vec))


def input_to_img(vec: torch.Tensor) -> PIL.Image:
    # Generate an image
    with torch.no_grad():
        output = stack.gan.generator(vec, truncation=truncation)

    # If you have a GPU put back on CPU
    output = output.to('cpu')

    imgs = convert_to_images(output)
    return imgs[0]
