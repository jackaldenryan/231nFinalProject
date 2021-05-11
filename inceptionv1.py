import torchvision
model = torchvision.models.googlenet(pretrained=True).to("cuda")
model.eval()
