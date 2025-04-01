import os
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from PIL import Image

class HARImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}
        
        for idx, class_name in enumerate(sorted(os.listdir(root_dir))):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                self.class_to_idx[class_name] = idx
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(idx)
        
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label

class HARDataModule(LightningDataModule):
    def __init__(self, data_dir="train", batch_size=32, num_workers=4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup(self, stage=None):
        dataset = HARImageDataset(self.data_dir, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = (len(dataset) - train_size) // 2
        test_size = len(dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

class HARResNet50(LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
    
    def test_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = self.criterion(outputs, labels)
        acc = (outputs.argmax(dim=1) == labels).float().mean()
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == "__main__":
    data_module = HARDataModule("train", batch_size=64)
    num_classes = len(os.listdir("train"))
    model = HARResNet50(num_classes)

    checkpoint_callback = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1, dirpath="checkpoints/", filename="best_model")
    early_stopping_callback = EarlyStopping(monitor="val_acc", mode="max", patience=5)

    trainer = Trainer(max_epochs=20, accelerator="gpu" if torch.cuda.is_available() else "cpu",
                       callbacks=[checkpoint_callback, early_stopping_callback], num_sanity_val_steps=1)
    trainer.fit(model, datamodule=data_module)

    torch.save(model.state_dict(), "har_resnet50.pth")

    trainer.test(model, datamodule=data_module)

    # def predict_image(model_path, image_path, class_names):
        # model = HARResNet50(len(class_names))
        # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # model.eval()
        
        # transform = transforms.Compose([
        #     transforms.Resize((224, 224)),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        
        # image = Image.open(image_path).convert("RGB")
        # image = transform(image).unsqueeze(0)
        
        # with torch.no_grad():
        #     output = model(image)
        #     predicted_class = output.argmax(dim=1).item()
        
        # return class_names[predicted_class]
    
    # class_names = sorted(os.listdir("train"))
    # result = predict_image("checkpoints/best_model.ckpt", "test.jpg", class_names)
    # print("Predicted action:", result)
