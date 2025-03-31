"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a CNN model to predict dislocation coordinates and their probability of presence

"""


import torch
import torch.nn as nn
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json
from torchvision import transforms
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import joblib 
from PIL import Image
#from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.init as init


# ==============================
# 1 Dataset construction
# ==============================
class CustomDataset(Dataset):
    def __init__(self, root, json_file, transform=None):
        self.root = root  # Image path
        self.json_file = json_file  # Path to JSON file
        self.transform = transform  # Transformations to be applied
        self.labels_data = self.load_json_data()  # Charger les données JSON
        self.image_groups = self.group_image_files()  # Groupes d'images à fusionner
        self.base_names = list(self.image_groups.keys())  # Noms de base pour accès facile
        
        # Create a scaler to transform labels in the range [0, 1]
        self.scaler = MinMaxScaler(feature_range=(0, 1))

        # Accumulate all positions
        all_positions = []
        for base_name in self.base_names:
            dislocations = self.labels_data[base_name]["dislocations"]  # On utilise le nom de base
            positions = []
            for dislocation in dislocations:
                if "x" in dislocation and "y" in dislocation:
                    positions.append([dislocation["x"], dislocation["y"]])
            all_positions.extend(positions)
        
        # Convert to numpy table and apply fit
        all_positions = torch.tensor(all_positions, dtype=torch.float32).numpy()  
        self.scaler.fit(all_positions)  # Fit dataset

        # Saving the scaler after you've made fit
        scaler_path = os.path.join(self.root, "scaler.pkl")
        joblib.dump(self.scaler, scaler_path)

    def group_image_files(self):
        """Group images by their base name (image_0, image_1, etc.)"""
        valid_extensions = ('.png', '.jpg', '.jpeg')
        image_files = [f for f in os.listdir(self.root) if f.lower().endswith(valid_extensions)]
        
        # Create groups of images
        image_groups = {}
        for img_file in image_files:
            # Split the filename to get base name
            parts = img_file.split('_')
            base_name = f"{'_'.join(parts[:-1])}.{img_file.split('.')[-1]}"  # Combine all parts except the last
            
            if base_name not in image_groups:
                image_groups[base_name] = []
            
            image_groups[base_name].append(img_file)
        
        # Sort each group by channel number (1, 2, 3)
        for base_name in image_groups:
            image_groups[base_name].sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        return image_groups
    
    def load_json_data(self):
        """Loads the JSON file and returns the data"""
        with open(self.json_file, 'r') as f:
            return json.load(f)
    
    def __len__(self):
        """Returns total number of image groups"""
        return len(self.base_names)
    
    def __getitem__(self, idx):
        """Returns a stacked image and its labels"""
        # Get base name and associated image files
        base_name = self.base_names[idx]
        image_files = self.image_groups[base_name]
        
        # Load and stack all channels
        channels = []
        for img_file in image_files:
            img_path = os.path.join(self.root, img_file)
            img = Image.open(img_path).convert("L")  # Convert to grayscale
            # Application des transformations (incluant ToTensor)
            if self.transform:
                img = self.transform(img)  # Shape: (1, 20, 20)
                
    
            channels.append(img)

        # Fusion des canaux
        stacked = torch.cat(channels, dim=0)  # Shape: (3, 20, 20)
        
        # Get labels using the base name
        dislocations = self.labels_data[base_name]["dislocations"]
        
        # Prepare labels (positions and probabilities)
        positions = []
        probabilities = []
        
        for dislocation in dislocations:
            if "x" in dislocation and "y" in dislocation:
                positions.append([dislocation["x"], dislocation["y"]])
                probabilities.append([dislocation["p1"], dislocation["p0"]])
        
        # Transform positions in the interval (0,1) and convert them into tensors
        positions = np.array(positions, dtype=np.float32)     
        positions = self.scaler.transform(positions)
        positions = torch.tensor(positions, dtype=torch.float32) 
        probabilities = torch.tensor(probabilities, dtype=torch.float32)

        
        return stacked, positions, probabilities, base_name


# ==============================
# 2 Padded dataset
# ==============================
'''def custom_collate(batch):
    """
    Function to manage variable sizes in a data batch.
    It groups images normally, and applies padding to positions and probabilities.
    """
    # Stack images normally (make sure images are already in tensor form)
    images = torch.stack([item[0] for item in batch])  
    
    # Keep positions and probabilities as tensor lists
    positions = [item[1] for item in batch]
    probabilities = [item[2] for item in batch]

    # Extract image names 
    image_names = [item[3] for item in batch]  
    
    # Apply padding to positions and probabilities
    # Sequences are padded to have the same length (batch_first=True, so that it doesn't invert the dimensions of the tensor)
    padded_positions = pad_sequence(positions, batch_first=True, padding_value=0.0)  # Padding on positions
    padded_probabilities = pad_sequence(probabilities, batch_first=True, padding_value=0.0)  # Padding on probabilités

    

    # Create a mask to ignore padded elements (1 for real, 0 for padded)
    positions_mask = (padded_positions.sum(dim=-1) != 0).float()  # Mask to ignore padded positions
    probabilities_mask = (padded_probabilities.sum(dim=-1) != 0).float()  # Mask to ignore padded probabilities


    return images, padded_positions, padded_probabilities, positions_mask, probabilities_mask, image_names'''



# ==============================
# 3️ Definition of the CNN Model
# ==============================
'''
def initialize_weights_he(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)  # Initialiser le biais à zéro '''

class CNN(nn.Module):
    def __init__(self, num_classes, num_coordinates):
        super(CNN, self).__init__()
        
        
        self.num_classes = num_classes  # Number of probability classes
        self. num_coordinates =  num_coordinates  # Number of coordinates

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding=1)
        

        
        #self.bn1 = nn.BatchNorm2d(64)  # BatchNorm after conv layer
        #self.bn2 = nn.BatchNorm2d(128)
        #self.bn3 = nn.BatchNorm2d(256) 
        #self.bn4 = nn.BatchNorm2d(512)
               
        
        # Activation layer
        self.relu = nn.ReLU()
        #self.sigm = nn.Sigmoid()  


        # Dropout 
        #self.dropout = nn.Dropout(0.3)  
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.avpool = nn.AvgPool2d(kernel_size=2, stride=2)


        # Fully connected layers
        flattened_size = 512 * 3 * 3 # Size after convolutions

        #self.fc1 = nn.Linear(flattened_size, 512)
        #self.fc2 = nn.Linear(512, 256)
        #self.fc3 = nn.Linear(512, 256)
        #self.fc4 = nn.Linear(256, 128)
        #self.fc5 = nn.Linear(128, 128)
    


        #print(f"max_dislocations_in_batch: {self.max_dislocations_in_batch}")

        # Classification output variable (num_classes = 2 (p0, p1), for each dislocation) 
        self.classification_head = nn.Linear(flattened_size, self.num_classes)  

        # Regression output variable (num_coordinates = 2 (x, y), for each dislocation)
        self.regression_head = nn.Linear(flattened_size, self.num_coordinates)  # output = torch.tensor([[x1_1, y1_1, x2_1, y2_1, x3_1, y3_1]])  # 1 image, 3 dislocations, 2 coordinates (x, y) per dislocation


        # Appliquer He Initialization
        #self.apply(initialize_weights_he)

        
    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        
        x = x.view(x.size(0), -1)  # Flatten

        #x = self.dropout(x)
        #x_fc2 = self.relu(self.bn7(self.fc2(x_fc1)))
        #x_fc3 = self.relu(self.bn8(self.fc3(x_fc2)))
        #x_fc = self.relu(self.bn9(self.fc4(x_fc)))
        #x_fc = self.relu(self.bn10(self.fc5(x_fc)))


        #print(x.shape)

        
        # Classification output 
        classification_logits = self.classification_head(x) 
        
        #Adapt the size to max_dislocations_in_batch (Batch, max_dislocations, 2)
        classification_logits = classification_logits.view(-1, self.num_classes)

        #classification_probs = F.softmax(classification_logits, dim=-1)  # Just for display

        # Regression output 
        regression_output = self.regression_head(x) #self.sigm(

        #Adapt the size to max_dislocations_in_batch (Batch, max_dislocations, 2)
        regression_output = regression_output.view(-1, self.num_coordinates) # output = torch.tensor([[x1_1, y1_1], [x2_1, y2_1], [x3_1, y3_1]])  # 1 image, 3 dislocations, 2 coordonnées (x, y) par dislocation

        return classification_logits, regression_output




# ==============================
# 4 Training function
# ==============================
def train_model(model, train_loader, val_loader, criterion_classification, criterion_regression, optimizer, device, num_epochs, alpha, beta, save_dir_model, scheduler=None):
    """
    Model training function with validation every 2 epochs.

    Args:
        model: The model to be train
        train_loader: DataLoader for training
        val_loader: DataLoader for validation
        criterion_classification: Loss function for classification
        criterion_regression: Loss function for regression
        optimizer: Optimiseur
        device: "cuda" ou "cpu"
        num_epochs: Total number of epochs
        scheduler: (Optionnal) Scheduler for adjust learning rate

    Returns:
        Dictionary containing training and validation metrics
    """

    # Lists for storing metrics
    metrics = {
        "Epochs": [],
        "train_loss_classification": [],
        "train_loss_regression": [],
        "train_loss_total": [],
        "train_accuracy_classification": [],
        "train_accuracy_regression": [],
        "train_accuracy_total": [],
        "val_loss_classification": [],
        "val_loss_regression": [],
        "val_loss_total": [],
        "val_accuracy_classification": [],
        "val_accuracy_regression": [],
        "val_accuracy_total": [],
    }

    best_val_acc = 0  # Track the best validation accuracy


    for epoch in range(num_epochs):
        model.train()  # Training mode
        
        running_loss_classification = 0.0
        running_loss_regression = 0.0
        running_loss_total = 0.0
        correct_classification = 0
        correct_regression = 0
        total = 0

        for images, positions, probabilities, _ in train_loader:
            # Sending data to the GPU/CPU
            images = images.to(device)
            positions = positions.to(device)
            probabilities = probabilities.to(device)

            optimizer.zero_grad()  # Reset gradients

            # Forward pass
            classification_logits, regression_output = model(images)

            # Flatten logits : [batch_size, num_classes]
            classification_logits = classification_logits #.reshape(-1, 2)  
            
            # Transform padded_probabilities into class indices and flatten targets : [batch_size * max_dislocations]
            target_classes = probabilities.argmax(dim=-1).view(-1) 
            
            #print(target_classes)

            positions = positions.reshape(-1, 2) 

            #print(classification_logits, target_classes, positions, regression_output)


            # Loss calculation
            loss_classification = criterion_classification(classification_logits, target_classes)
            loss_regression = criterion_regression(regression_output, positions)


            # Sum of loss
            total_loss =  alpha * loss_regression + beta * loss_classification 

            # Backpropagation and weight updating
            total_loss.backward()
            optimizer.step()

            # Metrics update
            running_loss_classification += loss_classification.item()
            running_loss_regression += loss_regression.item()
            running_loss_total += total_loss.item()

            # Calculation of classification accuracy
            _, predicted_classes = torch.max(classification_logits, 1) 

            correct_classification += (predicted_classes == target_classes).sum().item() 

            #target_labels = target_classes.argmax(axis=1)  # Convertit [0,1] en 1 et [1,0] en 0
            #predicted_labels = predicted_classes.argmax(axis=1)  # Même conversion pour les prédictions
            #correct_classification += f1_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='binary')

            # Calculation of regression accuracy (eg: distance < seuil)

            #print(np.shape(regression_output), np.shape(positions))

            

            correct_regression += (torch.abs(regression_output - positions) <= 1/20).sum().item()


            total += target_classes.size(0)

            #print(total, correct_regression)


            #if scheduler:
             #   scheduler.step()
            


        # Dynamic adjustment of alpha and beta weights after each epoch
        with torch.no_grad():
            class_loss = running_loss_classification / len(train_loader)
            regress_loss = running_loss_regression / len(train_loader)
        
            alpha = (class_loss + 1e-6) / (regress_loss + class_loss + 2 * 1e-6)
            beta = (regress_loss + 1e-6) / (regress_loss + class_loss + 2 * 1e-6)  # Increase the weight beta of the classification
            

        if (epoch + 1) % 1 == 0:
            # Average losses and accuracies for the entire train_loader
            metrics["Epochs"].append(epoch + 1)
            metrics["train_loss_classification"].append(running_loss_classification / len(train_loader))
            metrics["train_loss_regression"].append(running_loss_regression / len(train_loader))
            metrics["train_loss_total"].append(running_loss_total / len(train_loader))
            metrics["train_accuracy_classification"].append(100 * correct_classification / total ) 
            metrics["train_accuracy_regression"].append(100 * correct_regression / (total * regression_output.shape[-1]))  # Per feature x and y (regression_output.shape[-1]=2)
            metrics["train_accuracy_total"].append((metrics["train_accuracy_classification"][-1] + metrics["train_accuracy_regression"][-1]) / 2)

            
            # Display training stats
            print(f"Époque {epoch + 1}/{num_epochs} \n "
                f"  ###### TRAIN ##### \n"
                f"Loss classification: {metrics['train_loss_classification'][-1]:.4f}\n"
                f"Loss regression: {metrics['train_loss_regression'][-1]:.4f}\n"
                f"Total loss: {metrics['train_loss_total'][-1]:.4f}\n"
                f"Accuracy classification: {metrics['train_accuracy_classification'][-1]:.2f}%\n"
                f"Accuracy regression: {metrics['train_accuracy_regression'][-1]:.2f}%\n"
                f"Total accuracy: {metrics['train_accuracy_total'][-1]:.2f}%")

        

        # ==============================
        # 4 Validation every 2 epochs
        # ==============================

        if (epoch + 1) % 1 == 0:
            val_metrics = validate_model(model, val_loader, criterion_classification, criterion_regression, alpha, beta, device)
            for key in val_metrics:
                metrics[key].append(val_metrics[key])


            print(f"  ###### VALIDATION ##### \n"
                  f"Loss classification: {metrics['val_loss_classification'][-1]:.4f} \n"
                  f"Loss regression: {metrics['val_loss_regression'][-1]:.4f} \n"
                  f"Total loss: {metrics['val_loss_total'][-1]:.4f} \n"
                  f"Accuracy classification: {metrics['val_accuracy_classification'][-1]:.2f}% \n"
                  f"Accuracy regression: {metrics['val_accuracy_regression'][-1]:.2f}% \n"
                  f"Total Accuracy: {metrics['val_accuracy_total'][-1]:.2f}%")
            
            # Save the model if validation loss improves
            os.makedirs(save_dir_model, exist_ok=True)  # Creates the folder if it does not exist
            save_path_best_model = os.path.join(save_dir_model, "Best_Model_CNN.pth")
   
            if metrics["val_accuracy_regression"][-1] > best_val_acc:
                best_val_acc = metrics["val_accuracy_regression"][-1]
                torch.save(model.state_dict(), save_path_best_model)
                print(f"Best model saved with Val Accuracy for regression: {best_val_acc:.4f}") 

        # Apply scheduler if defined
           
        if scheduler:
            scheduler.step(metrics["val_accuracy_total"][-1])
            print("Learning rate:", scheduler.get_last_lr())



    return alpha, beta, metrics



# ==============================
# 5 Validation function
# ==============================
def validate_model(model, val_loader, criterion_classification, criterion_regression, alpha, beta, device):
    """
    Model validation function.

    Returns:
        Dictionary containing losses and accuracies """

    model.eval()  # Evaluation mode
    val_loss_classification = 0.0
    val_loss_regression = 0.0
    val_loss_total = 0.0
    correct_classification = 0
    correct_regression = 0
    total = 0

    with torch.no_grad(): # Disable gradient calculation during validation
        for images, positions, probabilities, _ in val_loader:
            images = images.to(device)
            positions = positions.to(device)
            probabilities = probabilities.to(device)

            classification_logits, regression_output = model(images)

            classification_logits = classification_logits  #.reshape(-1, 2)
            target_classes = probabilities.argmax(dim=-1).view(-1) 

            positions = positions.reshape(-1, 2) 
            

            loss_classification = criterion_classification(classification_logits, target_classes)
            loss_regression = criterion_regression(regression_output, positions)

            
            total_loss = alpha * loss_regression + beta * loss_classification 

            val_loss_classification += loss_classification.item()
            val_loss_regression += loss_regression.item()
            val_loss_total += total_loss.item()

            # Calculation of classification accuracy
            _, predicted_classes = torch.max(classification_logits, 1) # For CE

            correct_classification += (predicted_classes == target_classes).sum().item()

            #target_labels = target_classes.argmax(axis=1)  # Convertit [0,1] en 1 et [1,0] en 0
            #predicted_labels = predicted_classes.argmax(axis=1)  # Même conversion pour les prédictions
            #correct_classification += f1_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='binary')

            correct_regression += (torch.abs(regression_output - positions) <= 1/20).sum().item()



            total += target_classes.size(0)

    return {
        "val_loss_classification": val_loss_classification / len(val_loader),
        "val_loss_regression": val_loss_regression / len(val_loader),
        "val_loss_total": val_loss_total / len(val_loader),
        "val_accuracy_classification": 100 * correct_classification / total,
        "val_accuracy_regression": 100 * correct_regression / (total * regression_output.shape[-1]),
        "val_accuracy_total": (100 * correct_classification / total + 100 * correct_regression / (total * regression_output.shape[-1])) / 2
    }


# ==============================
# 6 Plot function
# ==============================
def plot_loss_accuracy(metrics, save_dir):
    """
    Function to create and save a graph showing the evolution of losses and accuracies for training and validation.

    Args:
        metrics (dict): Dictionary containing training and validation metrics.
        save_dir (str): Directory in which to save the graphic image.
    Returns:
        str: Path to saved file.
    """

    # Creating data for training
    df_train = pd.DataFrame({
        "Epoch": metrics["Epochs"]* 6,  # Repeat epochs for all three loss/accuracy types
        "Loss_Accuracy": metrics["train_loss_classification"] + metrics["train_loss_regression"] + metrics["train_loss_total"] +
                        metrics["train_accuracy_classification"] + metrics["train_accuracy_regression"] + metrics["train_accuracy_total"],
        "Type": (["Loss_classification"] * len(metrics["train_loss_classification"]) + 
                 ["Loss_regression"] * len(metrics["train_loss_regression"]) + 
                 ["Loss_total"] * len(metrics["train_loss_total"]) + 
                 ["Accuracy_classification"] * len(metrics["train_accuracy_classification"]) + 
                 ["Accuracy_regression"] * len(metrics["train_accuracy_regression"]) + 
                 ["Accuracy_total"] * len(metrics["train_accuracy_total"])),
        "Dataset": ["Train"] * len(metrics["train_loss_classification"])*6
    })

    # Creating data for validation
    df_val = pd.DataFrame({
        "Epoch": metrics["Epochs"]* 6,
        "Loss_Accuracy": metrics["val_loss_classification"] + metrics["val_loss_regression"] + metrics["val_loss_total"] +
                         metrics["val_accuracy_classification"] + metrics["val_accuracy_regression"] + metrics["val_accuracy_total"],
        "Type": (["Loss_classification"] * len(metrics["val_loss_classification"]) + 
                 ["Loss_regression"] * len(metrics["val_loss_regression"]) + 
                 ["Loss_total"] * len(metrics["val_loss_total"]) + 
                 ["Accuracy_classification"] * len(metrics["val_accuracy_classification"]) + 
                 ["Accuracy_regression"] * len(metrics["val_accuracy_regression"]) + 
                 ["Accuracy_total"] * len(metrics["val_accuracy_total"])),
        "Dataset": ["Validation"] * len(metrics["val_loss_classification"])*6
    })


    # Combining DataFrames
    df = pd.concat([df_train, df_val], ignore_index=True) # ignore_index=True: is used to reset the index

    # Seaborn configuration
    sns.set_theme(style="darkgrid")

    # Creation of the figure
    plt.figure(figsize=(18, 10))

    # Figure for training
    for i, loss_type in enumerate(["Loss_classification", "Loss_regression", "Loss_total"]):
        plt.subplot(2, 6, i + 1)
        sns.lineplot(x="Epoch", y="Loss_Accuracy", hue="Type", data=df[(df["Dataset"] == "Train") & (df["Type"] == loss_type)], marker="o")
        plt.title(f"Train - {loss_type.replace('_', ' ').title()} ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

    for i, accuracy_type in enumerate(["Accuracy_classification", "Accuracy_regression", "Accuracy_total"]):
        plt.subplot(2, 6, i + 4)
        sns.lineplot(x="Epoch", y="Loss_Accuracy", hue="Type", data=df[(df["Dataset"] == "Train") & (df["Type"] == accuracy_type)], marker="o")
        plt.title(f"Train - {accuracy_type.replace('_', ' ').title()} ")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")

    # Figure for validation
    for i, loss_type in enumerate(["Loss_classification", "Loss_regression", "Loss_total"]):
        plt.subplot(2, 6, i + 7)
        sns.lineplot(x="Epoch", y="Loss_Accuracy", hue="Type", data=df[(df["Dataset"] == "Validation") & (df["Type"] == loss_type)], marker="o")
        plt.title(f"Validation - {loss_type.replace('_', ' ').title()} ")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")

    for i, accuracy_type in enumerate(["Accuracy_classification", "Accuracy_regression", "Accuracy_total"]):
        plt.subplot(2, 6, i + 10)
        sns.lineplot(x="Epoch", y="Loss_Accuracy", hue="Type", data=df[(df["Dataset"] == "Validation") & (df["Type"] == accuracy_type)], marker="o")
        plt.title(f"Validation - {accuracy_type.replace('_', ' ').title()} ")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
   

    # Display adjustment
    plt.tight_layout()

    # Check if the folder exists, otherwise create it
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save figure
    Plot_file_path = os.path.join(save_dir, 'loss_accuracy_plot.png')
    plt.savefig(Plot_file_path)
    plt.close()

    return Plot_file_path


# ==============================
# 7 Test function
# ==============================
def test_model(model, test_loader, criterion_classification, criterion_regression, device, save_dir_prediction_true, alpha_f, beta_f, scaler_path):
    model.eval()  # Evaluation mode
    test_loss_classification = 0.0
    test_loss_regression = 0.0
    test_loss_total = 0.0
    correct_classification = 0
    correct_regression = 0
    total = 0
    
    predictions_true_list = []

    # Load scaler
    scaler = joblib.load(scaler_path)

    with torch.no_grad():  # Disable gradient calculation during testing
        for images, positions, probabilities, image_names in test_loader:
            
            images = images.to(device)
            positions = positions.to(device)
            probabilities = probabilities.to(device)
    
            
            # Forward pass
            classification_logits, regression_output = model(images)

            # Probabilities
            predicted_probs = F.softmax(classification_logits, dim=-1) 

            # Calculation of loss for classification
            classification_logits = classification_logits  #.reshape(-1, 2)
            target_classes = probabilities.argmax(dim=-1).view(-1) 

            positions = positions.reshape(-1, 2) 
            
            loss_classification = criterion_classification(classification_logits, target_classes)

            # Calculation of loss for regression
            loss_regression = criterion_regression(regression_output, positions)
            

            # Total loss
            total_loss = alpha_f * loss_regression + beta_f * loss_classification 


            test_loss_classification += loss_classification.item()
            test_loss_regression += loss_regression.item()
            test_loss_total += total_loss.item()

            # Calculation of classification accuracy
            _, predicted_classes = torch.max(classification_logits, 1) # For CE

            correct_classification += (predicted_classes == target_classes).sum().item()

            #target_labels = target_classes.argmax(axis=1)  # Convertit [0,1] en 1 et [1,0] en 0
            #predicted_labels = predicted_classes.argmax(axis=1)  # Même conversion pour les prédictions
            #correct_classification += f1_score(target_labels.cpu().numpy(), predicted_labels.cpu().numpy(), average='binary')

            correct_regression += (torch.abs(regression_output - positions) <= 1/20).sum().item()

        

            total += target_classes.size(0)

            # For displaying results in a JSON file

            # Convert to numpy to apply inverse_transform
            predicted_positions_np =  regression_output.cpu().numpy()
            true_positions_np = positions.cpu().numpy()

            # Inverse transform to recover the real values
            predicted_positions_real = np.array([scaler.inverse_transform(batch.reshape(1, -1)) for batch in predicted_positions_np])
            true_positions_real = np.array([scaler.inverse_transform(batch.reshape(1, -1)) for batch in true_positions_np])

            # Save results for each batch item
            for i in range(len(true_positions_real)):
                predictions_true_list.append({
                    "Image": image_names[i],
                    "Positions": {
                        "True":true_positions_real[i].tolist(),
                        "Predicted": predicted_positions_real[i].tolist()
                    },
                    "Probabilities": {
                        "True": probabilities[i].tolist(),
                        "Predicted": predicted_probs[i].tolist()
                    }
                })


    # Check if the folder exists, otherwise create it
    if not os.path.exists(save_dir_prediction_true):
        os.makedirs(save_dir_prediction_true)

    file_prediction_true = os.path.join(save_dir_prediction_true, 'Predictions_and_True.json')    
    # Save predictions to a JSON file
    with open(file_prediction_true, "w") as f:
        json.dump(predictions_true_list, f, indent=4)

    print(f"Predictions and true values ​​saved in {file_prediction_true}")


    #Save loss and accuracies for test
     
    Loss_accuracies = {
        "alpha" : alpha_f, 
        "beta" : beta_f,
        "Loss classification" : test_loss_classification / len(test_loader),
        "Loss regression" : test_loss_regression / len(test_loader),
        "Total loss" : test_loss_total / len(test_loader),
        "Classification accuracie in %" : 100 * correct_classification / total,
        "Regression accuracie in %" : 100 * correct_regression / (total * regression_output.shape[-1]),
        "Total accuracie in %" : (100 * correct_classification / total  + 100 * correct_regression / (total * regression_output.shape[-1])) / 2
    } 
     
    file_loss_accuracies = os.path.join(save_dir_prediction_true, 'Test_loss_accuracies.json')    
    with open(file_loss_accuracies, "w") as f:
        json.dump(Loss_accuracies, f, indent=4)

    print(f"Predictions and true values ​​saved in {file_loss_accuracies}")
    
    
    # Display results
    print(f"###### TEST ######\n")
    print(f"Loss classification : {test_loss_classification / len(test_loader):.4f}\n")
    print(f"Loss regression : {test_loss_regression / len(test_loader):.4f}\n")
    print(f"Total loss : {test_loss_total / len(test_loader):.4f}\n")
    print(f"Accuracy classification :  {100 * correct_classification / total :.2f}%\n")
    print(f"Accuracy regression : {100 * correct_regression / (total * regression_output.shape[-1]):.2f}%\n")
    print(f"Total accuracy : {(100 * correct_classification / total + 100 * correct_regression / (total * regression_output.shape[-1])) / 2:.2f}%") 

    return {
        "test_loss_classification": test_loss_classification / len(test_loader),
        "test_loss_regression": test_loss_regression / len(test_loader),
        "test_loss_total": total_loss / len(test_loader),
        "test_accuracy_classification":  100 * correct_classification / total,
        "test_accuracy_regression": 100 * correct_regression / (total * regression_output.shape[-1]),
        "test_accuracy_total": (100 * correct_classification / total + 100 * correct_regression / (total * regression_output.shape[-1])) / 2 
    }














