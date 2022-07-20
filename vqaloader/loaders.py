import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import time
import json

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
    IMAGE_PATH = {"train": "train_val_images/train_images", "val": "train_val_images/train_images", "test": "test_images"}
    
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
        assert split in ["train", "val", "test"]
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



########################################################################################
# VQAv2 https://visualqa.org/download.html
########################################################################################

def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)

class VQAv2Dataset(Dataset):
    IMAGE_PATH = {"train": ("train2014", "v2_OpenEnded_mscoco_train2014_questions.json", "v2_mscoco_train2014_annotations.json"), 
                  "val": ("val2014", "v2_OpenEnded_mscoco_val2014_questions.json", "v2_mscoco_val2014_annotations.json"),  
                  "testdev": ("test2015", "v2_OpenEnded_mscoco_test-dev2015_questions.json"), 
                  "test": ("test2015", "v2_OpenEnded_mscoco_test2015_questions.json")}
    
    def __init__(self, split, data_path="",
                 image_transforms=None, question_transforms=None, tokenize=None,
                 answer_selection=most_common_from_dict,
                 verbose=True, testing=False):
        """
        split train, val, test
        balanced True, False
        image_transforms
        question_transforms
        """
        start_time = time.time()
        self.split = split
        self.testing = testing
        self.answer_selection = answer_selection
        assert split in ["train", "val", "testdev", "test"]
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize

        if verbose:
            path = ""
            print(f"Start loading VQAv2 Dataset from {path}", flush=True)
        
        # Questions
        path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][1]))
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data["questions"])
        df["image_path"] = df["image_id"].apply(lambda x: f"{self.IMAGE_PATH[split][0]}/COCO_{self.IMAGE_PATH[split][0]}_{x:012d}.jpg")
        
        # Annotations
        if not testing:
            path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][2]))
            with open(path, 'r') as f:
                data = json.load(f)   
            df_annotations = pd.DataFrame(data["annotations"])
            df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='left')
            df["image_id"] = df["image_id_x"]
            if not all(df["image_id_y"] == df["image_id_x"]):
                print("There is something wrong with image_id")
            del df["image_id_x"]
            del df["image_id_y"]
        self.df = df
        self.n_samples = self.df.shape[0]
        if verbose:
            print(f"Loading VQAv2 Dataset done in {time.time()-start_time:.1f} seconds. Loaded {self.n_samples} samples.")
        
    def __getitem__(self, index):
        # image input
        image_id = self.df.iloc[index]["image_id"]
        image_path = self.df.iloc[index]["image_path"]
        question = self.df.iloc[index]["question"]
        question_id = self.df.iloc[index]["question_id"]
        split = self.split
        if not self.testing:
            main_answer = self.df.iloc[index]["multiple_choice_answer"] # Already extracted main answer
            answers = self.df.iloc[index]["answers"] # list of dicts: [{'answer': 'net', 'answer_confidence': 'maybe', 'answer_id': 1}, ...]
            selected_answers = self.answer_selection(self.df.iloc[index]["answers"]) # Apply answer_selection() function to list of dict
        
        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, image_path))        
        #print("image_path ---->", image_path)
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
            return {"img": img, "image_id": image_id, "question_id": question_id, "question": question}
        else:
            return {"img": img, "image_id": image_id, "question_id": question_id, "question": question, 
                    "main_answer": main_answer, "answers": answers, "answers": selected_answers}

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
    

########################################################################################
# OK-VQA https://okvqa.allenai.org/
########################################################################################

def most_common_from_dict(dct):
    lst = [x["answer"] for x in dct]
    return max(set(lst), key=lst.count)

class OKVQADataset(Dataset):
    IMAGE_PATH = {"train": ("train2014", "OpenEnded_mscoco_train2014_questions.json", "mscoco_train2014_annotations.json"), 
                  "test": ("val2014", "OpenEnded_mscoco_val2014_questions.json", "mscoco_val2014_annotations.json")}
    
    def __init__(self, split, data_path="",
                 image_transforms=None, question_transforms=None, tokenize=None,
                 answer_selection=most_common_from_dict,
                 verbose=True, testing=False):
        """
        split train, val, test
        balanced True, False
        image_transforms
        question_transforms
        """
        start_time = time.time()
        self.split = split
        self.testing = testing
        self.answer_selection = answer_selection
        assert split in ["train", "test"]
        self.data_path = data_path
        self.image_transforms = image_transforms
        self.question_transforms = question_transforms
        self.tokenize = tokenize

        if verbose:
            path = ""
            print(f"Start loading OKVQA Dataset from {path}", flush=True)
        
        # Questions
        path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][1]))
        with open(path, 'r') as f:
            data = json.load(f)
        df = pd.DataFrame(data["questions"])
        df["image_path"] = df["image_id"].apply(lambda x: f"{self.IMAGE_PATH[split][0]}/COCO_{self.IMAGE_PATH[split][0]}_{x:012d}.jpg")
        
        # Annotations
        if not testing:
            path = os.path.expanduser(os.path.join(data_path, self.IMAGE_PATH[split][2]))
            with open(path, 'r') as f:
                data = json.load(f)   
            df_annotations = pd.DataFrame(data["annotations"])
            df = pd.merge(df, df_annotations, left_on='question_id', right_on='question_id', how='left')
            # Check if image_id are still correct, remove newly created columns with x and y ending and just use the name image_id
            assert df["image_id_x"].tolist() == df["image_id_y"].tolist(), "image_id in df and df_annotations does not match."
            df["image_id"] = df["image_id_x"]
            del df["image_id_x"]
            del df["image_id_y"]
        self.df = df
        self.n_samples = self.df.shape[0]
        if verbose:
            print(f"Loading OKVQA Dataset done in {time.time()-start_time:.1f} seconds. Loaded {self.n_samples} samples.")
        
    def __getitem__(self, index):
        # image input
        image_id = self.df.iloc[index]["image_id"]
        image_path = self.df.iloc[index]["image_path"]
        # question input
        question_id = self.df.iloc[index]["question_id"]
        question = self.df.iloc[index]["question"]
        # answer and question type
        answer_type = self.df.iloc[index]["answer_type"]
        question_type = self.df.iloc[index]["question_type"]
        # split
        split = self.split
        # specify target if available (i.e. answer)
        if not self.testing:
            answer_list = self.df.iloc[index]["answers"] # Return whole list
            selected_answers = self.answer_selection(self.df.iloc[index]["answers"]) # Apply answer_selection() function to list of dict
        
        # Load and transform image
        image_path = os.path.expanduser(os.path.join(self.data_path, image_path))        
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
            return question_id, answer_list, selected_answers, img, question

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples
    
if __name__ == "__main__":    

    from vqaloader.loaders import GQADataset, TextVQADataset, VQAv2Dataset, OKVQADataset

    # GQADataset
    for split, testing in [("train", True), ("val", True),  ("testdev", True), ("test", True), ("challenge", True)]: #, ("submission", True)]:
        dataset = GQADataset(split=split, balanced=True, data_path="~/Data/GQA", testing=testing)
        print("Length:", len(dataset), "\nData:", dataset[0], "\n")
        

    # TextVQADataset
    for split, testing in [("train", True), ("val", True), ("test", True), ]:
        dataset = TextVQADataset(split=split, data_path="~/Data/TextVQA", testing=testing)
        print("Length:", len(dataset), "\nData:", dataset[0], "\n")


    # VQAv2Dataset
    for split, testing in [("train", True), ("val", True),  ("testdev", True), ("test", True)]:
        dataset = VQAv2Dataset(split=split, data_path="~/Data/VQAv2", testing=testing)
        print("Length:", len(dataset), "\nData:", dataset[0], "\n")

    # OKVQADataset
    for split, testing in [("train", True), ("test", True)]:  
        dataset = OKVQADataset(split=split,  data_path="~/Data/OKVQA", testing=testing)
        print("Length:", len(dataset), "\nData:", dataset[0], "\n")
