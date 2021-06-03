import shutil
import glob

i = 0
for file in glob.iglob("outputs/imagenet-activation-analysis/**/*.jpg", recursive=True):
    shutil.copy(
        file, "./outputs/imagenet-activation-analysis/onedir/" + str(i) + ".jpg")
    i += 1
