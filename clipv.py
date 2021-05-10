from clip import load

model, transform = load("RN50", device="cuda", jit=False)
