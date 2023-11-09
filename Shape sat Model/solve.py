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

colorStore = [c for c in COLORS if c not in  ["black","dark_grey","grey"]]
colorIterator = listIterator(colorStore)

def printNext(text):
    cprint(text,colorIterator.next())


model_data = len(os.listdir("data/train"))
print(f"Found {model_data} model choices.")

learning_rate = 0.025
learning_rate = 0.005


class stateSaver:
    def load(self,train=False):

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
mean = np.array([0.5, 0.5, 0.5])
std = np.array([0.25, 0.25, 0.25])

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ]),


}

data_dir = 'data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=64,
                                             shuffle=True, num_workers=32)
              for x in ['train', 'val']

               }
from PIL import Image
import io
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return data_transforms['val'](image).unsqueeze(0)
def transform_image_PIL(image):
    return data_transforms['val'](image).unsqueeze(0)





dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val',]}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print(class_names)


def imshow(inp, title):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.title(title)
    plt.show()


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

#imshow(out, title=[class_names[x] for x in classes])

def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs):
        printNext('Epoch {}/{}'.format(epoch, num_epochs - 1))
        printNext('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)


                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)




                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            printNext('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model



def getPrediction(model,object):
    i = object['input'].to(device)
    output = model(i)
    _, pred = torch.max(output, 1)
    object['label'] = class_names[pred.item()]
    return object


#### Finetuning the convnet ####
# Load a pretrained model and reset final fully connected layer.

saveModule = stateSaver(
    "shapesat.pth"
)

model,optimizer,epoch_start = saveModule.load(train=True)
print(f"Current epoch beginning at is: {epoch_start}")



#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, 7)
# Here the size of each output sample is set to 2.
# Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).


model = model.to(device)

criterion = nn.CrossEntropyLoss()

# Observe that all parameters are being optimized


# StepLR Decays the learning rate of each parameter group by gamma every step_size epochs
# Decay LR by a factor of 0.1 every 7 epochs
# Learning rate scheduling should be applied after optimizer’s update
# e.g., you should write your code this way:
# for epoch in range(100):
#     train(...)
#     validate(...)
#     scheduler.step()

step_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

epochSelected = 5



#input: {"path" , "tensor"}
#output: {"path", "label"}
model.eval()

for c in class_names:
    try:
        os.mkdir(f"ai_label/{c}")
    except: pass



if __name__ == "__main__":
    results = []
    i = 0
    total = len(os.listdir('split'))

    for FILENAME in os.listdir('split'):
        file = Image.open(f"split/{FILENAME}")
        this_tensor = transform_image_PIL(file)
        this = {"input": this_tensor, "Image": file, "filename": FILENAME}
        print(this_tensor)
        r = getPrediction(model, this)  # "input" :tensor , "Image": PIL
        break
        results.append(r)
        i += 1
        r['Image'].save(f"ai_label/{r['label']}/{r['filename']}")
        print(f"{i}/{total}. Saved {r['label']}")

