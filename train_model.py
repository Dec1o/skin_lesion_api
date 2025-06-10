import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from medmnist import INFO
import medmnist
import numpy as np

# 1) Definição da CNN (sem alterações)
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # Após duas poolings de 2: 28x28 -> 13x13 -> 5x5
        self.fc1   = nn.Linear(64 * 5 * 5, 128)
        self.fc2   = nn.Linear(128, num_classes)
        self.relu  = nn.ReLU()
        
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # (batch,32,13,13)
        x = self.pool(self.relu(self.conv2(x)))   # (batch,64,5,5)
        x = x.view(-1, 64 * 5 * 5)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2) Dataset customizado que extrai label escalar de dentro do array retornado pelo medmnist
class DermamnistDataset(Dataset):
    def __init__(self, medmnist_dataset):
        """
        medmnist_dataset: instância de medmnist.Dermamnist(split='...', download=True)
        """
        self.raw = medmnist_dataset

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, idx):
        item = self.raw[idx]
        img_pil = item[0]       # PIL.Image.Image
        label_arr = item[1]     # np.ndarray de shape (1,) ou escalar

        # Converter PIL -> NumPy array e permutar para (C,H,W)
        img_np = np.array(img_pil)          # shape (28,28,3), dtype=uint8
        img_np = img_np.transpose(2, 0, 1)   # agora (3,28,28)
        img_np = img_np.astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np)  # dtype float32

        # Extrair label escalar
        if isinstance(label_arr, np.ndarray):
            label_scalar = int(label_arr[0])
        else:
            # Caso já venha como escalar (pouco provável), só converte
            label_scalar = int(label_arr)
        label_tensor = torch.tensor(label_scalar, dtype=torch.long)

        return img_tensor, label_tensor


def main():
    # Detectar dispositivo (CPU ou GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando no dispositivo: {device}")

    # 3) Carregar o medmnist Dermamnist original
    data_flag = 'dermamnist'
    info      = INFO[data_flag]
    DataClass = getattr(medmnist, info['python_class'])

    # Baixa/Carrega splits de treino e teste
    train_raw = DataClass(split='train', download=True)
    test_raw  = DataClass(split='test',  download=True)

    print(f"Tamanho train_raw: {len(train_raw)}, test_raw: {len(test_raw)}")

    # 4) Envolver em Dataset customizado (que já faz a conversão correta)
    train_dataset = DermamnistDataset(train_raw)
    test_dataset  = DermamnistDataset(test_raw)

    # 5) DataLoader
    train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(dataset=test_dataset,  batch_size=32, shuffle=False)

    # 6) Instanciar modelo, loss e otimizador
    num_classes = len(info['label'])  # normalmente 7 classes
    model       = SimpleCNN(num_classes=num_classes).to(device)
    criterion   = nn.CrossEntropyLoss()
    optimizer   = optim.Adam(model.parameters(), lr=0.001)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)   # já é tensor float32, shape [batch,3,28,28]
            labels = labels.to(device)   # tensor long, shape [batch]

            optimizer.zero_grad()
            outputs = model(images)      # outputs: [batch, num_classes]
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}  |  Loss: {avg_loss:.4f}")

        # Validação simples no conjunto test
        model.eval()
        correct = 0
        total   = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"Val Accuracy: {acc:.2f}%\n")

    # 7) Salvar o modelo treinado
    torch.save(model.state_dict(), "lesion_model.pth")
    print("Modelo salvo em lesion_model.pth")


if __name__ == "__main__":
    main()
