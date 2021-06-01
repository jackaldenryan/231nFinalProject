from biggan import model as biggan
from inceptionv1 import model as inceptionv1
import torch
import torchvision
from pytorch_pretrained_biggan import (one_hot_from_names, truncated_noise_sample,
                                       convert_to_images)
import captum.optim as optimviz
import PIL
import numpy as np
from IPython.display import clear_output

# Fix the random seed
# torch.manual_seed(1337)
# np.random.seed(1337)
# random.seed(1337)

truncation = 1


class GANStack(torch.nn.Module):
    def __init__(self, gan, model):
        super(GANStack, self).__init__()
        self.gan = gan
        self.regularization = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(4),
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
        if self.steps % 2 == 0:
            pass  # display_optimized_image(x)
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


class ChannelActivationWithClassRegularization(torch.nn.Module):
    def __init__(self,
                 input_parametizer: SimpleInputParametizer,
                 target: torch.nn.Module, channel: int,
                 reg: float):
        super(ChannelActivationWithClassRegularization, self).__init__()
        self.reg = reg
        self.target = target
        self.channel = channel
        self.input_parametizer = input_parametizer
        self.loss_fn = optimviz.loss.ChannelActivation(target, channel)
        self.regularization = torch.nn.L1Loss()

    def forward(self, mappings):
        x = self.loss_fn(mappings)
        # Apply L2 regularization, but such that higher L2s are better
        x += self.reg * torch.sum(self.input_parametizer.class_vector ** 2)

        return x


def max_loss_summarize(loss_value: torch.Tensor):
    return -1 * loss_value.max()


def create_optimized_image(target: torch.nn.Module, channel: int, n_steps: int,
                           lr: float, reg: float,
                           verbose: bool = True) -> torch.Tensor:
    """
    :returns: (1, 256) input ready for biggan insertion, loss_history
    """
    input = SimpleInputParametizer()
    loss_fn = ChannelActivationWithClassRegularization(
        input, target, channel, reg)
    io = optimviz.InputOptimization(
        stack, loss_fn, input, torch.nn.Identity())
    print("Running optim with lr", lr, "reg", reg)
    history = io.optimize(optimviz.optimization.n_steps(
        n_steps, verbose), optimizer=torch.optim.Adam(io.parameters(), lr=lr))  # , loss_summarize_fn=max_loss_summarize)

    print("L2 of class vector =",
          torch.sum(input.class_vector ** 2))

    return input.forward(), history, input.class_vector.detach()


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
