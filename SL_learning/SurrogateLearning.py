import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader, TensorDataset
import joblib
import numpy as np
import os
import wandb

class DataPreparation:
    """Prepare actuation-to-tip-position samples for the DDKS regressor."""

    def __init__(self, data_path):
        data = pd.read_csv(data_path, header=None, names=["l1", "l2", "l3", "x", "y", "z"])

        self.X = data[["l1", "l2", "l3"]].values
        self.y = data[["x", "y", "z"]].values

        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.X = self.scaler_X.fit_transform(self.X)
        self.y = self.scaler_y.fit_transform(self.y)

        # Fixed split seed keeps the surrogate validation set reproducible.
        self.X_train, X_temp, self.y_train, y_temp = train_test_split(self.X, self.y, test_size=0.30, random_state=42)
        self.X_val, self.X_test, self.y_val, self.y_test = train_test_split(X_temp, y_temp, test_size=0.33, random_state=42)

        self.X_train = torch.tensor(self.X_train, dtype=torch.float32)
        self.y_train = torch.tensor(self.y_train, dtype=torch.float32)
        self.X_val = torch.tensor(self.X_val, dtype=torch.float32)
        self.y_val = torch.tensor(self.y_val, dtype=torch.float32)
        self.X_test = torch.tensor(self.X_test, dtype=torch.float32)
        self.y_test = torch.tensor(self.y_test, dtype=torch.float32)

    def get_data_loaders(self, batch_size=64):
        train_dataset = TensorDataset(self.X_train, self.y_train)
        val_dataset = TensorDataset(self.X_val, self.y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader

class NeuralNetwork(nn.Module):
    """MLP forward kinematic surrogate: encoded tendons -> Cartesian tip position."""

    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(3, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 3)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        
        return x


def main():
    
    wandb.init(project="Supervised learining", name="Best_(1)", config={"epochs": 300, "batch_size": 64, "learning_rate": 0.0001})
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))
    
    dataset_path = os.path.join(project_root, "dataset","Dataset-Actions-Positions.txt")
    data_prep = DataPreparation(dataset_path)
    train_loader, val_loader = data_prep.get_data_loaders(batch_size=64)

    model = NeuralNetwork()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)   # for lr = 0.0003 -> avg_acc = 0.2913
                                                            
    train_losses = []
    val_losses = []

    num_epochs = 300
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            predictions = model(X_batch)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for X_batch, y_batch in val_loader:
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        wandb.log({"epoch": epoch, "train_loss": avg_epoch_loss, "val_loss": avg_val_loss})

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Training Loss: {avg_epoch_loss}, Validation Loss: {avg_val_loss}")

    last_20_percent_epochs = int(num_epochs * 0.2)
    avg_val_loss_last_20_percent = sum(val_losses[-last_20_percent_epochs:]) / last_20_percent_epochs
    print(f"Average Validation Loss for the last 20% epochs: {avg_val_loss_last_20_percent}")

    model.eval()
    with torch.no_grad():
        test_predictions = model(data_prep.X_test)
        test_loss = criterion(test_predictions, data_prep.y_test)
        print(f"Final Test Loss: {test_loss.item()}")

        test_predictions = data_prep.scaler_y.inverse_transform(test_predictions.numpy())
        y_test = data_prep.scaler_y.inverse_transform(data_prep.y_test.numpy())

        r2 = r2_score(y_test, test_predictions)
        mae = mean_absolute_error(y_test, test_predictions)
        rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        print(f"R² Score: {r2:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    
    
    output_dir = os.path.join(project_root, "SurrogateModel")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_path = os.path.join(output_dir, "trained_model_sl_1.pth")
    scaler_X_path = os.path.join(output_dir, "scaler_X_sl_1.pkl")
    scaler_y_path = os.path.join(output_dir, "scaler_y_sl_1.pkl")

    torch.save(model.state_dict(), model_path)
    joblib.dump(data_prep.scaler_X, scaler_X_path)
    joblib.dump(data_prep.scaler_y, scaler_y_path)
    print(f"Model and scalers were saved to '{output_dir}'.")
    
if __name__ == "__main__":
    main()
