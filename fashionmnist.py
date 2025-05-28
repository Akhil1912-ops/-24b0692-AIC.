import os
import pandas as pd  # to read the csv files
import numpy as np  # to use numpy array
from PIL import Image  # to load image and check when converted
from tqdm import tqdm  # to check the progress bars
import torch  # pytorch
from torch import nn, optim  # to use nn and optimizers
# to load data and also split data
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms  # to import resnet50



DATA_DIR = "."  # stores the path of csv files
TRAIN_CSV = "fashion-mnist_train.csv"  # storing training file to train_csv
TEST_CSV = "fashion-mnist_test.csv"  # storing training file to test_csv
BATCH_SIZE = 64  # breaking the wholes set of pixels to multiples sets
NUM_CLASSES = 10  # there are 10 no of classes
NUM_EPOCHS_HEAD = 5  # 5 would be better number to not let model be overfit
NUM_EPOCHS_FINETUNE = 5  # would be better number to not let model be overfit
LR_HEAD = 1e-3  # took adivise from internet
LR_FINE = 1e-4  # took adivise from internet
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# creating a custom dataset
class FashionMNISTCSV(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        # sepearating first labels and pixels
        self.labels = self.df.iloc[:, 0].values.astype(np.int64)
        self.pixels = self.df.iloc[:, 1:].values.astype(np.uint8)
        self.transform = transform
      # defining a funtion to get to know its length

    def __len__(self):
        return len(self.labels)

     # a funtion which converts the csv file pixels data to 28,28 tensors
    def __getitem__(self, idx):
        img = self.pixels[idx].reshape(28, 28)
        img = Image.fromarray(img, mode="L")  # grayscale
        if self.transform:
            img = self.transform(img)
        label = self.labels[idx]
        return img, label


# convert 1to3 channels & resize to 224×224
common_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    # standertizing to allign with resnet50
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

# training data is sent to fasionmnist class and then trasformed to standers that a resnet50 want
train_dataset = FashionMNISTCSV("fashion-mnist_train.csv", transform=common_tf)
# calculating 10% of training dataset
n_val = int(len(train_dataset) * 0.1)
# calculating remaing 90% of dta set
n_train = len(train_dataset) - n_val
# spliting training DS into two parts training(90%) and validation(10%)
train_ds, val_ds = random_split(
    train_dataset, [n_train, n_val],
    generator=torch.Generator().manual_seed(42)
)
# loading both DS and spliting into batch size of 64
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=4)

# loading test data and diving into diffrent batches of size 64
test_ds = FashionMNISTCSV(os.path.join(
    DATA_DIR, TEST_CSV), transform=common_tf)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=4)


#  MODEL SETUP
def build_model(num_classes, freeze_backbone=True):
   # loading resnet50 model
    model = models.resnet50(pretrained=True)
    # frezing all the nn as its already trained
    if freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False
    # replace the head
    # defines how many layers does the final layer of resnet50 outputs
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),  # converting no of inputs to 256
        nn.ReLU(inplace=True),  # its a activation funtion
        nn.Dropout(0.5),  # turns off 50% of inputs to avoid overfitting
        nn.Linear(256, num_classes)  # converting no of outputs to 10
    )
    return model.to(DEVICE)


# training
def train_one_epoch(model, loader, criterion, optimizer):
    # setting model to training mode
    model.train()
    running_loss = 0.0
    correct = 0
    # using tqdm which shows the traning vizualizes clearly
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()  # clearing gradeints
        out = model(imgs)  # sending the images to model
        # caliculating the loss based on the label_predicted and actual label
        loss = criterion(out, labels)
        loss.backward()  # applying autograd to parameters again
        optimizer.step()  # modifing parameter(weights and bais) to something more sutable

        running_loss += loss.item() * imgs.size(0)  # calculating total loss
        preds = out.argmax(dim=1)  # predicts what is the class of images
        # adds to then list if the prediction is correct
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc


@torch.no_grad()  # now testing on validation data which we splitted to 10%
def validate(model, loader, criterion):
    model.eval()  # setting model to eval so than it dont not perform some function of testing
    running_loss = 0.0
    correct = 0
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        loss = criterion(out, labels)
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)
    return epoch_loss, epoch_acc


def main():
    # head training
    # sending inputs to model
    model = build_model(NUM_CLASSES, freeze_backbone=True)
    criterion = nn.CrossEntropyLoss()  # calcuilates the loss
    optimizer = optim.Adam(model.fc.parameters(),
                           lr=LR_HEAD, weight_decay=1e-4)  # adjust the weight and bais according to loss

    print("=== Training head only ===")
    for epoch in range(1, NUM_EPOCHS_HEAD+1):
        tr_loss, tr_acc = train_one_epoch(
            # using the function we have made earlier to calculate loss and accuracy
            model, train_loader, criterion, optimizer)
        # using the function we have made earlier to calculate loss and accuracy
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"[Head] Epoch {epoch}/{NUM_EPOCHS_HEAD}  "
              f"Train loss={tr_loss:.4f}, acc={tr_acc:.4f}  "
              f"Val loss={val_loss:.4f}, acc={val_acc:.4f}")

    # fine-tuning: unfreeze last block & head
    print("\n=== Fine-tuning last ResNet block + head ===")
    # unfreezing layer4 & fc
    for name, p in model.named_parameters():
        # specifing both layer4 and head
        if name.startswith("layer4") or name.startswith("fc"):
            p.requires_grad = True  # turing on autograd for both of them

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                           # defing another optimizer for finetuning recommeded  by chatgtp
                           lr=LR_FINE, weight_decay=1e-5)

    for epoch in range(1, NUM_EPOCHS_FINETUNE+1):
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimizer)  # using the model to finetuning also
        val_loss, val_acc = validate(model, val_loader, criterion)
        print(f"[Fine] Epoch {epoch}/{NUM_EPOCHS_FINETUNE}  "
              f"Train loss={tr_loss:.4f}, acc={tr_acc:.4f}  "
              f"Val loss={val_loss:.4f}, acc={val_acc:.4f}")

    # final test evaluation
    print("\n=== Testing ===")
    # using validate funtion because there is no need to optimize here
    test_loss, test_acc = validate(model, test_loader, criterion)
    print(f"Test loss={test_loss:.4f}, acc={test_acc:.4f}")

    # save  model
    torch.save(model.state_dict(), "fashion_resnet50_ft.pth")
    print("Model saved to fashion_resnet50_ft.pth")


if __name__ == "__main__":
    main()
