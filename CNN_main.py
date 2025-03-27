"""
@author: Dr Yen Fred WOGUEM 

@description: This script trains a CNN model to predict dislocation coordinates and their probability of presence

"""

 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split
import os
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR
from CNN_utilities import CustomDataset, CNN
from CNN_utilities import train_model, plot_loss_accuracy, test_model
from torch_lr_finder import LRFinder
import matplotlib.pyplot as plt



start_time = datetime.now()  # Start timer



# Define device for calculations
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

# ==============================
# 1 Hyperparamètres
# ==============================
batch_size = 32          # 16 32 64 Number of images per batch
learning_rate = 1e-5       # Learning rate
weight_decay = 1e-3        # L2 regularisation 1e-2 
num_epochs =  100        # Number of epochs


# ==============================
# 2 Load data 
# ==============================

# Load JSON file
json_file = "/home/woguem/Bureau/Projet_Machine_Learning/GB_Cu_001_Generation/Generated_Images/Grain_Boundary.json"  # Chemin vers le fichier JSON contenant les labels


#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), Normalisation RGB pour avoir des valeur entre (-1,1)

transform = transforms.Compose([transforms.Resize((20, 20)), transforms.ToTensor()]) # transforms.Resize((100, 100)), convert the image into a PyTorch tensor whose pixels are now between (0,1)


root = "/home/woguem/Bureau/Projet_Machine_Learning/GB_Cu_001_Generation/Generated_Images"  # Path to image folder


dataset = CustomDataset(root=root, json_file=json_file, transform=transform)



# ========================================================
# 3 Split data into training, validation and testing
# ========================================================

train_size = int(0.7 * len(dataset))  # 60% for training
val_size = int(0.2 * len(dataset))   # 20% for validation
test_size = len(dataset) - train_size - val_size  # The rest for the test


train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])



train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=32, shuffle=True) #collate_fn=custom_collate,
val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=32, shuffle=False)

print(len(train_loader))
print(len(val_loader))
print(len(test_loader))  


# ========================================================
# 4 Displaying loaded data
# ========================================================


'''
print(f"Nombre d'images : {len(dataset)}") # Data set and image size

#image = dataset[0]  # This gives the first image after transformation
#print(f"Index image 0 : {image}")

# Batch recovery
batch = next(iter(train_loader))

print(batch)

# Breaking down the batch
images = batch[0]  # Images
positions = batch[1]  # Positions 
probabilities = batch[2]  # Probabilities 


# Display the shapes of the various elements
print("Images:", images.shape)  # (Batch_size, Channels, Height, Width)
print("Positions:", positions.shape)  # (Batch_size, max_dislocations, 2) - coordonnées x, y
print("Probabilities:", probabilities.shape)  # (Batch_size, max_dislocations, 2) - p1, p0


# Show some images
fig, axes = plt.subplots(8, 4, figsize=(10, 5)) #4, 8
axes = axes.flatten()
for img, ax in zip(images[:32], axes):
    img = img.numpy().transpose((1, 2, 0))  # Convert to (H, W, C) format
    ax.imshow(img)
    ax.axis("off")

plt.savefig("Image_batch")   

#plt.show() 
'''



# ==============================
# 5 Model initialization
# ==============================

num_classes = 2     # Number of classes (p1, p0)
num_coordinates = 2 # Number of coordinates (x, y)


model = CNN(num_classes, num_coordinates).to(device)



criterion_classification = nn.CrossEntropyLoss() # Loss function for classification (For CE)
criterion_regression = nn.SmoothL1Loss()         # Loss function for regression
optimizer = optim.Adam(model.parameters(), lr=learning_rate) #, weight_decay=weight_decay)     # Oprimizer for updating parameters (weights)
#optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9) #, weight_decay=1e-4)

# Scheduler : if used, reduces learning rate 
#steps_per_epoch = len(train_loader)
#scheduler = OneCycleLR(optimizer, max_lr=0.03, steps_per_epoch=steps_per_epoch, epochs=num_epochs)
#scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.9)


# ==============================
# 6 Training and Validation
# ==============================

# Fitting parameters based on classification and regression loss
alpha = 1
beta = 1

# Define the folder directory for saving the best model
save_dir_model = "/home/woguem/Bureau/Projet_Machine_Learning/GB_Cu_001_Generation/Best_Models_models"

# Calls the training function, which in turn calls the validation function

alpha_f, beta_f, metrics = train_model( model, train_loader, val_loader, 
                                        criterion_classification, criterion_regression, 
                                        optimizer, device, num_epochs, alpha, beta, save_dir_model, scheduler = None
)



#print(alpha_f, beta_f)

#Display lenght of outpout from training function 
#for key in metrics:
 #   print(f"{key}: {len(metrics[key])}")




# ====================================================================================================================================================
# 7 Test function output display (Loss, Accuracy, True positions, True probabilities, Predicted positions, Predicted Probabilities)
# ====================================================================================================================================================

# Specify the directory where you want to save the image
save_dir = "/home/woguem/Bureau/Projet_Machine_Learning/GB_Cu_001_Generation/Loss_Accuracy_Plots"  # Remplace par le chemin du répertoire de ton choix

Plot_file_path = plot_loss_accuracy(metrics, save_dir)

print(f"figure saved in  : {Plot_file_path}")


# =====================================================================================
# 8 Perform a test and record the predictions and true of positions and probabilities
# =====================================================================================


save_dir_prediction_true = "/home/woguem/Bureau/Projet_Machine_Learning/GB_Cu_001_Generation/Predictions_and_True"  # Remplace par le chemin du répertoire de ton choix

scaler_path = os.path.join(root, "scaler.pkl")

# Calling up the test function
test_metrics = test_model(model, test_loader, criterion_classification, criterion_regression, 
                          device, save_dir_prediction_true, alpha_f, beta_f, scaler_path )









end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")


















