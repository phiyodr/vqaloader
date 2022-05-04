# VQALoader

## Install

```bash
pip install git+https://github.com/phiyodr/vqaloader
```


## Data download


#### [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)

```bash
# cd ~/data/ # or anywhere you want to place it
mkdir GQA && cd GQA
ẁget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip # 20.3 GB images
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip # 1.4 GB questions
unzip images.zip
unzip questions1.2.zip
``````

#### [TextVQA](https://textvqa.org/dataset/)

* `train`
* `val`
* `test`

One answer per question.

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

#### [VQAv2](https://visualqa.org/download.html)

* `train`
* `val`
* `testdev`
* `test`



```bash
# cd ~/data/ # or anywhere you want to place it
mkdir VQAv2 && cd VQAv2
# VQA Input Questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip #  443,757 questions 
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip #  214,354 questions 
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip #  447,793 questions
v2_OpenEnded_mscoco_test-dev2015_questions.json 
# VQA Input Images
wget http://images.cocodataset.org/zips/train2014.zip # 82,783 images 
wget http://images.cocodataset.org/zips/val2014.zip # 40,504 images 
wget http://images.cocodataset.org/zips/test2015.zip # 81,434 images 
# Answers
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip
# unzip
unzip "*.zip"
``````

#### [OK-VQA](https://okvqa.allenai.org/download.html)

* `train` for training
* `test` for testing. Authors call the file "val" and state it is for testing. 


10 answers per question with "anwwer_id", "raw_answer" (we use this), "answer_confidence", "answer" (stemmed).

```bash
# cd ~/data/ # or anywhere you want to place it
mkdir OKVQA && cd OKVQA
# VQA Input Questions
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json # Training questions
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json # Testing questions
# VQA Input Images
wget http://images.cocodataset.org/zips/train2014.zip # Training images
wget http://images.cocodataset.org/zips/val2014.zip # Testing images
# Answers
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json # (v1.1 updated 7/29/2020)
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json #(v1.1 updated 7/29/2020)
# unzip
unzip "*.zip"
echo "Done!"
``````



## Usage

* GQADataset

```python
from vqaloader.loaders import GQADataset
dataset = GQADataset(split="train", balanced=True, data_path="~/Data/GQA", testing=False)
print(dataset[0])
```

* TextVQADataset

```python
from vqaloader.loaders import TextVQADataset
dataset = TextVQADataset(split="train", data_path="~/Data/TextVQA", testing=False)
print(dataset[0])
```


* VQAv2Dataset

```python
from vqaloader.loaders import VQAv2Dataset
dataset = VQAv2Dataset(split="train", data_path="~/Data/VQAv2", testing=False)
print(dataset[0])
```

* OKVQADataset

 ```python  
from vqaloader.loaders import OKVQADataset 
dataset = OKVQADataset(split="train",  data_path="~/Data/OKVQA", testing=False)
print(dataset[0])  
```