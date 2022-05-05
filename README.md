# VQALoader

## Install

```bash
pip install git+https://github.com/phiyodr/vqaloader
```


## Data download


#### [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)

* Splits: `challenge`, `submission`, `test`, `testdev`, `train`, `val`
* df columns: `['semantic', 'entailed', 'equivalent', 'question', 'imageId',
       'isBalanced', 'groups', 'answer', 'semanticStr', 'annotations', 'types',
       'fullAnswer'`
* Answer: One answer in `answer`.


```bash
# cd ~/data/ # or anywhere you want to place it
mkdir GQA && cd GQA
ẁget https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip       # Download Images (20.3 GB images)
wget https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip # Download Questions (1.4 GB questions)
unzip images.zip
unzip questions1.2.zip
``````

#### [TextVQA](https://textvqa.org/dataset/)

* Splits: `train`, `val`, `test`
* df columns: `['dataset_type', 'dataset_name', 'dataset_version', 'data']`
* Answer: List of 10 answers in `answers`.


```bash
# cd ~/data/ # or anywhere you want to place it
mkdir TextVQA && cd TextVQA
ẁget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_train.json # Training set 34,602 questions (103 MB)
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json   # Validation set 5,000 questions (16MB)
wget https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_test.json  # Test set 5,734 questions (13MB)
wget https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip   # Training set 21,953 images (6.6 GB)
wget https://dl.fbaipublicfiles.com/textvqa/images/test_images.zip        # Test set 3,289 images (926MB)
unzip train_val_images.zip
unzip test_images.zip
``````

#### [VQAv2](https://visualqa.org/download.html)

* Splits: `train`, `val`, `testdev`, `test`
* df columns: `['question', 'question_id', 'image_path', 'question_type', 'multiple_choice_answer', 'answers', 'answer_type', 'image_id']`
* Answers:
    * One main answer in `multiple_choice_answer`.
    * List of 10 dicts in `answers`. Example: `{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1}`.


```bash
# cd ~/data/ # or anywhere you want to place it
mkdir VQAv2 && cd VQAv2
# VQA Input Questions
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip # Training questions 2017 v2.0* 443,757 questions 
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip   # Validation questions 2017 v2.0* 214,354 questions 
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Test_mscoco.zip  # Testing questions 2017 v2.0 447,793 questions 
v2_OpenEnded_mscoco_test-dev2015_questions.json 
# VQA Input Images
wget http://images.cocodataset.org/zips/train2014.zip # COCO Training images 82,783 images 
wget http://images.cocodataset.org/zips/val2014.zip   # Validation images 40,504 images 
wget http://images.cocodataset.org/zips/test2015.zip  # Testing images 81,434 images 
# VQA Annotations Balanced Real Images
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip # Training annotations 2017 v2.0* 4,437,570 answers 
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip   # Validation annotations 2017 v2.0* 2,143,540 answers 
# unzip
unzip "*.zip"
``````

#### [OK-VQA](https://okvqa.allenai.org/download.html)

* Splits:
    * `train` for training
    * `test` for testing. Authors call the file "val" and state it is for testing. 
* df columns: `['question', 'question_id', 'image_path', 'answer_type', 'question_type',
       'answers', 'image_id']`
* Answers: List of 10 dicts in `answers`. Example: `{'answer_id': 8, 'raw_answer': 'labrador retriever', 'answer_confidence': 'yes', 'answer': 'labrador retriev'}` (`raw_answer` is stemmed, `answer` is full answer).

```bash
# cd ~/data/ # or anywhere you want to place it
mkdir OKVQA && cd OKVQA
# OK-VQA Input Questions
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_train2014_questions.json # Training questions
wget https://okvqa.allenai.org/static/data/OpenEnded_mscoco_val2014_questions.json   # Testing questions
# Images (COCO)
wget http://images.cocodataset.org/zips/train2014.zip # Training images
wget http://images.cocodataset.org/zips/val2014.zip   # Testing images
# OK-VQA Annotations
wget https://okvqa.allenai.org/static/data/mscoco_train2014_annotations.json # Training annotations (v1.1 updated 7/29/2020)
wget https://okvqa.allenai.org/static/data/mscoco_val2014_annotations.json   # Testing annotations (v1.1 updated 7/29/2020)
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