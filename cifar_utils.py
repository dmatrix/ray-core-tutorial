
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from ray import train
import ray.train.torch
from ray.data.extensions import TensorArray
from ray.air import session
from ray.train.torch import TorchCheckpoint
from ray.serve.http_adapters import NdArray

CLASSES = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    # this is the reason for normalizing (with mean, std) for each RGB channel
    # Normalization helps reduce or skewing and helps with faster CNN training
    # https://discuss.pytorch.org/t/understanding-transform-normalize/21730
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

def train_dataset_factory():
    """
    Download the train CiFAR 10 dataset into the root dir
    """
    return torchvision.datasets.CIFAR10(root="~/data", 
                                        download=True, 
                                        train=True,
                                        transform=transform)

def test_dataset_factory():
    """
    Download the test CiFAR 10 dataset into the root dir
    """
    return torchvision.datasets.CIFAR10(root="~/data", 
                                        download=True, 
                                        train=False, 
                                        transform=transform)

def convert_batch_to_numpy(batch: Tuple[torch.Tensor, int]) -> Dict[str, np.ndarray]:
    images = np.array([image.numpy() for image, _ in batch])
    labels = np.array([label for _, label in batch])
    return {"image": images, "label": labels}

class BasicBlock(nn.Module):
    expansion = 1
    

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()

        DROPOUT = 0.1

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout(DROPOUT)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes),
                nn.Dropout(DROPOUT)
            )

    def forward(self, x):
        out = F.relu(self.dropout(self.bn1(self.conv1(x))))
        out = self.dropout(self.bn2(self.conv2(out)))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.log_softmax(out, dim=-1)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def train_loop_per_worker(config):
    device = (ray.train.torch.get_device() if torch.cuda.is_available() else torch.device("cpu")
    )
    
    # Prepare model for training. This involves moving the 
    # model to appropriate device, putting into the training mode
    model = train.torch.prepare_model(ResNet18())

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"], momentum=0.9)

    # Fetch training Ray dataset from the session; this is automatically
    # distributed to the right worker node's process where the training
    # run
    train_dataset_shard = session.get_dataset_shard("train")

    # Iterate over epochs
    epochs = config.get("epochs", 10)
    batch_size = config.get("batch_size", 32)
    verbose = config.get("verbose", 0)
    lr = config.get("lr", 0.01)
    
        
    for epoch in tqdm.tqdm(range(epochs)):
        if verbose:
            print(f"Training epoch:{epoch+1}/{epochs} | batch_size:{batch_size} | lr:{lr}")
        
        train_loss = 0.0
        train_loss = 0.0
        total_images = 0
        
        # loop over batches for each epoch
        train_dataset_batches = train_dataset_shard.iter_torch_batches(
            batch_size=batch_size,
        )
        for i, batch in enumerate(train_dataset_batches):
            # Get the inputs and labels
            inputs, labels = batch["image"], batch["label"]
            num_images = inputs.shape[0]
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * num_images
            total_images += num_images

        train_loss /= total_images
        metrics = dict(train_loss=train_loss)
        
        # Create a Torch checkpoint from the models state dictionary after each
        # epoch and report the metrics 
        checkpoint = TorchCheckpoint.from_state_dict(model.module.state_dict())
        session.report(metrics, checkpoint=checkpoint)

def convert_logits_to_classes(df):
    best_class = df["predictions"].map(lambda x: x.argmax())
    df["prediction"] = best_class
    return df
    
def calculate_prediction_scores(df):
    df["correct"] = df["prediction"] == df["label"]
    return df[["prediction", "label", "correct"]]

def json_to_numpy(payload: NdArray) -> pd.DataFrame:
      """Accepts an NdArray JSON from an HTTP body and converts it to a Numpy Array."""
      # Have to explicitly convert to float since np.array reads as a double.
      arr = np.array(payload.array, dtype=np.float32)
      return arr

def to_prediction_cls(lst):
    max_value = max(lst)
    idx = lst.index(max_value)
    cls = CLASSES[idx] 
    return idx,cls

def img_show(img):
    img = img / 2 + 0.5     # unnormalize
    # npimg = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()