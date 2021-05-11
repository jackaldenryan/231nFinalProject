import ray
import requests

ray.init(num_cpus=6, ignore_reinit_error=True)

@ray.remote
def download_url(url, name):
	response = requests.get(url)
	file = open(name, "wb")
	file.write(response.content)
	file.close()
​
# yahoo flickr creative commons
# ray.get([download_url.remote(f'https://oaiggoh.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/contrastive_v2/lucid.dataset_examples/_dataset_examples/dataset%3Dyfcc%26op%3Dimage_block_4%252F2%252Fadd_5%253A0/channel_{i}_40.png', f'channel_{i}.png') for i in range(2048)])
​
# imagenet
ray.get([download_url.remote(f'https://oaiggoh.blob.core.windows.net/microscopeprod/2020-07-25/2020-07-25/contrastive_v2/lucid.dataset_examples/_dataset_examples/dataset%3Dimagenet%26op%3Dimage_block_4%252F2%252Fadd_5%253A0/channel_{i}_40.png', f'channel_{i}.png') for i in range(2048)])