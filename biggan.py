from pytorch_pretrained_biggan import BigGAN
model = BigGAN.from_pretrained('biggan-deep-256').to("cuda")