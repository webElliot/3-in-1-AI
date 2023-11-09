import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from termcolor import cprint,COLORS
from iterator import *
from PIL import Image
import io


class stateSaver:
    def load(self,train,learning_rate,model_data):

        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = model.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
        loaded=True
        try:
            checkpoint = torch.load(self.path)
            cprint("Success! loaded from file.", "green")
        except:
            cprint("Failed to load from file.","red")
            loaded = False
        model.fc = nn.Linear(num_ftrs, model_data)
        if loaded:
            model.load_state_dict(checkpoint['model_state'])

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

        if loaded:
            optimizer.load_state_dict(checkpoint['optim_state'])
            epoch = checkpoint['epoch']
        else:
            epoch=0

        if train:
            model.train()
            cprint("Loaded model Mode: Training!", "green")
        else:
            model.eval()
            cprint("Loaded model Mode: Evaluation!", "green")

        return model , optimizer , epoch

    def save(self,epoch,model,optimizer):
        checkpoint = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optim_state": optimizer.state_dict()
        }
        torch.save(checkpoint, self.path)



    def __init__(self,path):
        self.path = path

class sittingAi:
    def transform_image(self,image_bytes):
        image = Image.open(io.BytesIO(image_bytes))
        return self.data_transforms['val'](image).unsqueeze(0)

    def transform_image_PIL(self,image):
        return self.data_transforms['val'](image).unsqueeze(0)
    def __init__(self):
        self.model_data = len(os.listdir("data/train"))
        print(f"Found {self.model_data} model choices.")
        self.learning_rate = 0.025

        self.mean = np.array([0.5, 0.5, 0.5])
        self.std = np.array([0.25, 0.25, 0.25])

        self.data_transforms = {
            'train': transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std)
            ]),

        }

        self.data_dir = 'data'
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(self.data_dir, x), self.data_transforms[x])
                          for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=64,
                                                      shuffle=True, num_workers=32)
                       for x in ['train', 'val']

                       }

        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val', ]}
        self.class_names = self.image_datasets['train'].classes
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.class_names)

        self.saveModule = stateSaver(
            "rotation.pth"
        )

        model, optimizer, epoch_start = self.saveModule.load(train=False,learning_rate=self.learning_rate,model_data=self.model_data)
        print(f"Current epoch beginning at is: {epoch_start}")
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        epochSelected = 5
        model.eval()
        self.model = model

    def run(self):
        results = []
        i = 0
        total = len(os.listdir('split'))

        for FILENAME in os.listdir('split'):
            file = Image.open(f"split/{FILENAME}")
            this_tensor = self.transform_image_PIL(file)
            this = {"input": this_tensor, "Image": file, "filename": FILENAME}
            r = self.getPrediction(this)  # "input" :tensor , "Image": PIL
            results.append(r)
            i += 1
            r['Image'].save(f"ai_label/{r['label']}/{r['filename']}")
            print(f"{i}/{total}. Saved {r['label']}")

    def getPrediction(self, object):
        i = object['input'].to(self.device)
        output = self.model(i)
        _, pred = torch.max(output, 1)
        object['label'] = self.class_names[pred.item()]
        return object





