import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def process_image(image_path):
    ''' 
    Scales, crops, and normalizes a PIL image for a PyTorch       
    model, returns an Numpy array
    '''
    # Open the image    
    
    img = Image.open(image_path)
    # Resize
    if img.size[0] > img.size[1]:
        img.thumbnail((10000, 256))
    else:
        img.thumbnail((256, 10000))
    # Crop 
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    
    return img

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


def predict(image_path, model, top_num=3, gpu=False, cat_to_name=None):
    
    # Process image
    img = process_image(image_path)
    
    if torch.cuda.is_available() and gpu:
        device = torch.device("cuda")
        image_tensor = torch.from_numpy(img).type(torch.cuda.FloatTensor)
        print("running in cuda . . .")
    else:
        device = torch.device("cpu")
        image_tensor = torch.from_numpy(img).type(torch.FloatTensor)
        print("running in cpu . . .")
        
    model.to(device)
    
    # Add batch of size 1 to image
    model_input = image_tensor.unsqueeze(0)
    
    # Probs
    probs = torch.exp(model.forward(model_input))
    
    # Top probs
    top_probs, top_labs = probs.topk(top_num)
    top_probs = top_probs.detach().cpu().tolist()[0] 
    top_labs = top_labs.detach().cpu().tolist()[0]
    
    # Convert indices to classes
    idx_to_class = {val: key for key, val in    
                                      model.class_to_idx.items()}
    
    top_labels = [idx_to_class[lab] for lab in top_labs]
    if cat_to_name is not None:
        top_names = [cat_to_name[idx_to_class[lab]] for lab in top_labs]
    else:
        top_names = top_labels
    return top_probs, top_labels, top_names