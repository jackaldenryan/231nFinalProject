import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
resnet18.eval()
resnet18 = resnet18.to("cuda")
