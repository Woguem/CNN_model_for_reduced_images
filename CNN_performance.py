"""
@author: Dr Yen Fred WOGUEM 

@description: This script uses a CNN model trained on one dislocation to test its performance on an image with many dislocations

"""

 

import torch
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
from scipy.ndimage import sobel


start_time = datetime.now()  # Start timer



class CNN(nn.Module):
    def __init__(self, num_classes=2, num_coordinates=2):
        super(CNN, self).__init__()
        
        
        self.num_classes = num_classes  # Number of probability classes
        self. num_coordinates =  num_coordinates  # Number of coordinates

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=2, padding=1)              
        
        # Activation layer
        self.relu = nn.ReLU()
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #self.avpool = nn.AvgPool2d(kernel_size=2, stride=2)


        # Fully connected layers
        flattened_size = 512 * 3 * 3 # Size after convolutions

        # Classification output variable (num_classes = 2 (p0, p1), for each dislocation) 
        self.classification_head = nn.Linear(flattened_size, self.num_classes)  

        # Regression output variable (num_coordinates = 2 (x, y), for each dislocation)
        self.regression_head = nn.Linear(flattened_size, self.num_coordinates)  # output = torch.tensor([[x1_1, y1_1, x2_1, y2_1, x3_1, y3_1]])  # 1 image, 3 dislocations, 2 coordinates (x, y) per dislocation

        
    def forward(self, x):

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        
        x = x.view(x.size(0), -1)  # Flatten

        # Classification output 
        classification_logits = self.classification_head(x) 
        
        #Adapt the size to max_dislocations_in_batch (Batch, max_dislocations, 2)
        classification_logits = classification_logits.view(-1, self.num_classes)

        #classification_probs = F.softmax(classification_logits, dim=-1)  # Just for display

        # Regression output 
        regression_output = self.regression_head(x) #self.sigm(

        #Adapt the size to max_dislocations_in_batch (Batch, max_dislocations, 2)
        regression_output = regression_output.view(-1, self.num_coordinates) 

        return classification_logits, regression_output


window_size = 20
stride = 10
threshold = 0.5


def sliding_window_inference(model, image, window_size=window_size, stride=stride, classification_threshold=threshold):
    c, h, w = image.shape  # Image (3 canaux)
    print(c, h, w)
    
    # Cartes de sortie
    classification_map = np.zeros((h, w))  # Stocke les scores de classification
    accumulator_map = np.zeros((h, w))  # Accumulateur pour compter les fenêtres
    regression_map = []  # Stocke les coordonnées prédites des dislocations

    for i in range(0, h - window_size + 1, stride):
        for j in range(0, w - window_size + 1, stride):
            patch = image[:, i:i + window_size, j:j + window_size]  # Extraire un patch (3, 20, 20)
            patch_tensor = torch.tensor(patch).unsqueeze(0).float()  # (1, 3, 20, 20)
            #print(patch_tensor , patch_tensor.shape)
            
            plt.imshow(patch.permute(1, 2, 0).numpy())
            plt.show()
            print(patch[0][13][12])

            quit()


            with torch.no_grad():
                regression_output, classification_logits = model(patch_tensor)  # Deux sorties du modèle

            #print(classification_logits)    
            classification_score = torch.sigmoid(classification_logits[0][0]).item()  # Convertir logits en probabilité
            
            # Ajouter le score de classification à la carte
            classification_map[i:i + window_size, j:j + window_size] += classification_score

            accumulator_map[i:i + window_size, j:j + window_size] += 1



            # Si le modèle détecte une dislocation (score > seuil), enregistrer la position
            if classification_score > classification_threshold:

                #print(classification_score)
                reg_coords = regression_output.numpy().flatten()  # Convertir en numpy
                x_pred = reg_coords[0]   + j + 10   # Convertir en coordonnée absolue
                y_pred = reg_coords[1]  + i + 10 
                print(x_pred, y_pred)
                regression_map.append((x_pred, y_pred))  # Ajouter la position prédite
            
            

    return classification_map, regression_map, accumulator_map

# Charger l’image et convertir en tenseur PyTorch
image = Image.open(r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\GB_Cu_001_Generation\Generated_Images\Image_0.png")

image_for_model = image.convert("L")



plot_image = image.convert("RGB")

transform = transforms.Compose([
    transforms.Resize((400, 400)),  # Redimensionner l'image si besoin
    transforms.ToTensor()  # Convertir en tenseur
])


plot_image = transform(plot_image)

image_tensor = transform(image_for_model)  # Résultat : Tensor (C, H, W)

print(image_tensor[:, 2].max(), image_tensor[:, 2].min(), image_tensor.shape)



channels = [image_tensor, image_tensor, image_tensor]
stacked = torch.cat(channels, dim=0)  # Shape: (3, 20, 20)

print(stacked.shape)

# Charger ton modèle pré-entraîné
model = CNN()
model.load_state_dict(torch.load(r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\GB_Cu_001_Generation\Best_Models_models\Best_Model_CNN.pth", map_location=torch.device('cpu')))  # Charger le modèle
model.eval()  # Mettre en mode évaluation

# Appliquer l'inférence
classification_map, regression_map, accumulator_map = sliding_window_inference(model, stacked)



c, h, w = stacked.shape

# Calcul de la moyenne
# Éviter la division par zéro en vérifiant l'accumulateur
with np.errstate(divide='ignore', invalid='ignore'):
    classification_map /= accumulator_map


classification_map = np.where(classification_map > threshold, classification_map, np.nan)


# Convertir en format (H, W, C) pour Matplotlib
image_np = stacked.permute(1, 2, 0).numpy()
plot_image = plot_image.permute(1, 2, 0).numpy()

# Afficher la carte de classification et de régression
plt.figure(figsize=(12, 6))

# Heatmap de la classification

window_size=window_size
stride=stride

plt.imshow(image_np)  # Image en fond

plt.show()
quit()
#plt.imshow(classification_map, cmap='gray', alpha=0.7)  # Heatmap superposée

colors = [
    'magenta',    # Magenta
    'cyan',      # Cyan
    'pink',      # Rose
    'green',     # Vert
    'purple',    # Violet
    'blue',      # Bleu
    'orange',    # Orange
    'brown',     # Marron
    'yellow',    # Jaune
    'red',       # Rouge
    'black'      # Black
]

k = 0

for i in range(0, h - window_size + 1, stride):
    for j in range(0, w - window_size + 1, stride):
        # Extraire la sous-région
        sub_region = classification_map[i:i + window_size, j:j + window_size]
        
        if not np.isnan(sub_region).any() :

            #print(sub_region)
            x_min, x_max = j, j + window_size
            y_min, y_max = i, i + window_size

            # Tracer les lignes horizontales
            plt.plot([x_min, x_max], [y_min, y_min], color=colors[k % len(colors)], linestyle='-', linewidth=0.5)
            plt.plot([x_min, x_max], [y_max, y_max], color=colors[k % len(colors)], linestyle='-', linewidth=0.5)

            # Tracer les lignes verticales
            plt.plot([x_min, x_min], [y_min, y_max], color=colors[k % len(colors)], linestyle='-', linewidth=0.5)
            plt.plot([x_max, x_max], [y_min, y_max], color=colors[k % len(colors)], linestyle='-', linewidth=0.5) 

            k+=1



l=0 
for (x, y) in regression_map:
    plt.scatter(x, y, color=colors[l % len(colors)], s=5)  # Points rouges pour les dislocations
    l+=1

#plt.ylim(250, 150)
plt.title("Classification and Regression Map (Dislocations Positions)")

plt.savefig(r"C:\Users\p09276\Post_doc_Yen_Fred\Projet_Machine_Learning_Julien\GB_Cu_001_Generation\Generated_Images\Dislocation_presence_and_positions_predictions_Image_0_stride_10.png", dpi=300)

#plt.show()







end_time = datetime.now()  # End of timer
execution_time = end_time - start_time
print(f"\nDurée d'exécution : {execution_time}")


















