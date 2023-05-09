# import resources
# %matplotlib inline

from PIL import Image
from io import BytesIO
# import matplotlib.pyplot as plt
# from matplotlib import cm
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

# get the "features" portion of VGG19 (we will not need the "classifier" portion)
vgg = models.vgg19(pretrained=True).features

# freeze all VGG parameters since we're only optimizing the target image
for param in vgg.parameters():
    param.requires_grad_(False)

# move the model to GPU, if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vgg.to(device)

def load_image(img_path, max_size=400, shape=None):
    ''' Load in and transform an image, making sure the image
       is <= 400 pixels in the x-y dims.'''
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')
    
    # large images will slow down processing
    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)
    
    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), 
                                             (0.229, 0.224, 0.225))])

    # discard the transparent, alpha channel (that's the :3) and add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

# load in content and style image
content_path = 'https://vignette.wikia.nocookie.net/lovecraft/images/c/cf/Screenshot_20171018-093500.jpg/revision/latest?cb=20171020174137'
content = load_image(
    content_path,
).to(device)

# Resize style to match content, makes code easier
style_path = 'https://d3d00swyhr67nd.cloudfront.net/w800h800/collection/SRY/RHU/SRY_RHU_THC0021-001.jpg'
style = load_image(
    style_path,
    shape=content.shape[-2:],
).to(device)

# helper function for un-normalizing an image 
# and converting it from a Tensor image to a NumPy image for display
def im_convert(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

# display the images
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
# content and style ims side-by-side
# ax1.imshow(im_convert(content))
# ax2.imshow(im_convert(style))

# print out VGG19 structure so you can see the names of various layers
print(vgg)

def get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    ## TODO: Complete mapping layer names of PyTorch's VGGNet to names from the paper
    ## Need the layers for the content and style representations of an image
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
        
    ## -- do not need to change the code below this line -- ##
    features = {}
    x = image
    # model._modules is a dictionary holding each module in the model
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    ## get the batch_size, depth, height, and width of the Tensor
    _, d, h, w = tensor.size()
    ## reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(d, h * w)
    ## calculate the gram matrix    
    gram = torch.mm(tensor, tensor.t())
    
    return gram

# get content and style features only once before forming the target image
content_features = get_features(content, vgg)
style_features = get_features(style, vgg)

# calculate the gram matrices for each layer of our style representation
style_grams = {
    layer: gram_matrix(style_features[layer]) for layer in style_features
}

# create a third "target" image and prep it for change
# it is a good idea to start off with the target as a copy of our *content* image
# then iteratively change its style
target = content.clone().requires_grad_(True).to(device)

# weights for each style layer 
# weighting earlier layers more will result in *larger* style artifacts
# notice we are excluding `conv4_2` our content representation
conv1_1 = 1.0
conv2_1 = 0.8
conv3_1 = 0.5
conv4_1 = 0.3
conv5_1 = 0.1
style_weights = {
    'conv1_1': conv1_1,
    'conv2_1': conv2_1,
    'conv3_1': conv3_1,
    'conv4_1': conv4_1,
    'conv5_1': conv5_1,
}

# you may choose to leave these as is
content_weight = 1  # alpha
style_weight = 1e6  # beta

# for displaying the target image, intermittently
show_every = 400

# iteration hyperparameters
learning_rate = 0.003
optimizer = optim.Adam(
    [target],
    lr=learning_rate,
)
steps = 5000  # decide how many iterations to update your image (5000)

min_loss = float("inf")
for ii in range(1, steps+1):
    # get the features target image and calculate the content loss
    target_features = get_features(target, vgg)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    
    # the style loss, initialized to 0
    style_loss = 0
    # iterate through each style layer and add to the style loss
    for layer in style_weights:
        # get the "target" style representation for the layer
        target_feature = target_features[layer]
        _, d, h, w = target_feature.shape
        
        # calculate the target gram matrix
        target_gram = gram_matrix(target_feature)
        
        # get the "style" style representation
        style_gram = style_grams[layer]
        # calculate the style loss for one layer, weighted appropriately
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        
        # add to the style loss
        style_loss += layer_style_loss / (d * h * w)
        
        
    # calculate the *total* loss
    total_loss = content_weight * content_loss + style_weight * style_loss
    
    ## -- do not need to change code, below -- ##
    # update your target image
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    min_loss = min(min_loss, total_loss.item())
    # display intermediate images and print the loss
    if  ii % show_every == 0:
        print(f'step: {ii}')
        print(f'loss: {min_loss}')
        # plt.imshow(im_convert(target))
        # plt.show()
print(f'step: {steps}')
print(f'loss: {min_loss}')

def img_from_np(array):
    return Image.fromarray(
        np.uint8(array*255),
        "RGB",
    )

initial_img = img_from_np(im_convert(content))
initial_img.save("initial_img.png")

final_img = img_from_np(im_convert(target))
final_img.save("final_img.png")