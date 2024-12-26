import re

import pandas as pd
from torch.utils.data import Dataset


class SmokingData(Dataset):
    def __init__(self, file_path, max_samples=100):
        self.data = pd.read_csv(file_path).head(max_samples)
        self.contexts = self.data["text"].tolist()
        self.questions = [
            "Given the fact that the following is an excerpt from a patient's doctor's note, "
            "is this patient a current smoker, past smoker, or has never smoked before?"
        ] * len(self.contexts)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {"context": self.contexts[idx], "question": self.questions[idx]}


class CancerData(Dataset):
    def __init__(self, file_path, max_samples=100):
        self.data = pd.read_csv(file_path).head(max_samples)
        self.contexts = []
        for text in self.data["text"]:
            sentences = re.findall(r"([^.]*?cancer[^.]*\.)", text)
            self.contexts.extend([sentence.lstrip() for sentence in sentences])
        self.questions = [
            "Given the fact that the following is an excerpt from a patient's doctor's note, "
            "what is the stage of cancer does the patient have?"
        ] * len(self.contexts)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return {"context": self.contexts[idx], "question": self.questions[idx]}


