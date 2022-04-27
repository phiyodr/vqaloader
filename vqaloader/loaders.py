import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time

########################################################################################
# GQA
########################################################################################

class GQADataset(Dataset):
    BALANCED_TYPE = {
        True: "balanced",
        False: "all"
    }
    def __init__(self, split, balanced=True, data_path="",
                 image_transforms=None, question_transforms=None, tokenize=None,
                 verbose=True, testing=False):
        """
        Args:
            split (str): Data split. One of ["challenge", "submission", "test", "testdev", "train", "val"]
            balanced (bool): You balanced version or full version.
            image_transforms:
            question_transforms:
            tokenize (fct):
            verbose (bool): Print some infos. Default=True
            testing (bool): Set to true for data splits without targets. Default=False.
        """
        start_time = time.time()
        self.split = split
        self.testing = testing
        assert split in ["challenge", "submission", "test", "testdev", "train", "val"]
        self.balanced = balanced
        self.balanced_type = self.BALANCED_TYPE[balanced]
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize
        
        if not balanced and split == "train":
            raise NotImplementedError
        else:
            self.file_name = f"questions1.2/{self.split}_{self.balanced_type}_questions.json"
            path = os.path.expanduser(os.path.join(data_path, self.file_name))
            if verbose:
                print(f"Start loading GQA Dataset from {path}", flush=True)
            self.df = pd.read_json(path, orient="index")       

        self.n_samples = self.df.shape[0]
        if verbose:
            print(f"Loading GQA Dataset done in {time.time()-start_time:.1f} seconds. Loaded {self.n_samples} samples.")
        
    def __getitem__(self, index):
        # image input
        sample_id = self.df.iloc[0].name
        image_id = self.df.iloc[index]["imageId"]
        question = self.df.iloc[index]["question"]
        split = self.split[index]
        if not self.testing:
            answer = self.df.iloc[index]["answer"]
            question_type = self.df.iloc[index]["groups"]["global"]
        
        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, "images", f"{image_id}.jpg"))        
        print(image_path)
        with open(image_path, "rb") as f:
            img = Image.open(f)
        if self.image_transforms:
            img = self.image_transforms(img)

        # Load, transform and tokenize question
        if self.question_transforms: 
            question = self.question_transforms(question)
        if self.tokenize:
            question = self.tokenize(question)

        # Return
        if self.testing:
            return {"sample_id": sample_id, "answer": None, "img": img, "question": question, "question_type": None}
        else:
            return {"sample_id": sample_id, "answer": answer, "img": img, "question": question, "question_type": question_type}

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    
########################################################################################
# TextVQA
########################################################################################

def cleaner(split):
    return "val" if split == "valid" else split

def most_common(lst):
    return max(set(lst), key=lst.count)

    
class TextVQADataset(Dataset):
    IMAGE_PATH = {"train": "train_val_images/train_images", "valid": "train_val_images/train_images", "test": "test_images"}
    
    def __init__(self, split, version="0.5.1", data_path="",
                 image_transforms=None, question_transforms=None, tokenize=None,
                 answer_selection=most_common,
                 verbose=True, testing=False):
        """
        split train, val, test
        balanced True, False
        image_transforms
        question_transforms
        """
        start_time = time.time()
        self.split = split
        self.version = version
        self.testing = testing
        self.answer_selection = answer_selection
        assert split in ["train", "valid", "test"]
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize

        self.file_name = f"{split}/TextVQA_{version}_{cleaner(split)}.json"

        path = os.path.expanduser(os.path.join(data_path, self.file_name))
        if verbose:
            print(f"Start loading TextVQA Dataset from {path}", flush=True)
        self.df = pd.read_json(path)       

        self.n_samples = self.df.shape[0]
        if verbose:
            print(f"Loading TextVQA Dataset done in {time.time()-start_time:.1f} seconds. Loaded {self.n_samples} samples.")
        
    def __getitem__(self, index):
        # image input
        image_id = self.df.iloc[index]["data"]["image_id"]
        question = self.df.iloc[index]["data"]["question"]
        question_id = self.df.iloc[index]["data"]["question_id"]
        split = self.df.iloc[index]["dataset_type"]
        if not self.testing:
            answers = self.df.iloc[index]["data"]["answers"]
            main_answer = self.answer_selection(self.df.iloc[index]["data"]["answers"])
        
        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, f"{self.IMAGE_PATH[self.split]}/{image_id}.jpg"))        
        print("image_path ---->", image_path)
        with open(image_path, "rb") as f:
            img = Image.open(f)
        if self.image_transforms:
            img = self.image_transforms(img)

        # Load, transform and tokenize question
        if self.question_transforms: 
            question = self.question_transforms(question)
        if self.tokenize:
            question = self.tokenize(question)

        # Return
        if self.testing:
            return question_id, img, question
        else:
            return question_id, answers, main_answer, img, question

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
if __name__ == '__main__':
    
    from utils import clip_tokenize
    print(clip_tokenize)

    print("===GQADataset===")
    dataset = GQADataset(split="train", balanced=True, data_path="~/Data/GQA", tokenize=clip_tokenize, testing=False)
    print(dataset[0])

    print("===TextVQADataset===")
    dataset = TextVQADataset(split="train",  data_path="~/Data/TextVQA", tokenize=clip_tokenize, testing=False)
    print(dataset[0])