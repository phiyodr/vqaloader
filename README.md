# VQALoader

## Install

```bash
pip install git+https://github.com/phiyodr/vqaloader
```


## Data download


* [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)

```bash
# cd ~/data/ # or anywhere you want to place it
mkdir GQA && cd GQA
ẁget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip # 20.3 GB images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip # 1.4 GB questions
unzip images.zip
unzip questions1.2.zip
``````

* [TextVQA](https://textvqa.org/dataset/)

```bash
# cd ~/data/ # or anywhere you want to place it
mkdir TextVQA && cd TextVQA
ẁget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json # 103 MB questions
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json # 16 MB questions
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json # 13 MB questions
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip # 6.6 GB images
wget https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip # 926 MB images
unzip train_val_images.zip
unzip test_images.zip
``````


## Usage

* GQADataset

```python
from vqaloader.loaders import GQADataset
dataset = GQADataset(split="train", balanced=True, data_path="~/data/GQA", testing=False)
print(dataset[0])

# With clip tokenizer
from vqaloader.utils import clip_tokenize
dataset = GQADataset(split="train", balanced=True, data_path="~/data/GQA", tokenize=clip_tokenize)
print(dataset[0])
```

* TextVQADataset

```python
from vqaloader.loaders import TextVQADataset
dataset = TextVQADataset(split="train", data_path="~/data/TextVQA", testing=False)
print(dataset[0])

# With clip tokenizer
from vqaloader.utils import clip_tokenize
dataset = TextVQADataset(split="train", data_path="~/data/TextVQA", tokenize=clip_tokenize)
print(dataset[0])
```