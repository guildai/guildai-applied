from PIL import Image
from io import BytesIO
import numpy as np

import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

def _get_vgg19_features():
    return models.vgg19(pretrained=True).features

def _freeze_vgg_parameters(vgg):
    for param in vgg.parameters():
        param.requires_grad_(False)

def _try_use_gpu_as_torch_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _load_image(img_path, max_size=400, shape=None):
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

    # discard the alpha channel, add the batch dimension
    image = in_transform(image)[:3,:,:].unsqueeze(0)
    return image

def _torch_tensor_to_np_image(tensor):
    """ Display a tensor as an image. """
    
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)

    return image

def _img_from_np_array(array):
    return Image.fromarray(
        np.uint8(array*255),
        "RGB",
    )

def _get_copy_of_content_image(content, device):
    return content.clone().requires_grad_(True).to(device)

def _get_features(image, model, layers=None):
    """ Run an image forward through a model and get the features for 
        a set of layers. Default layers are for VGGNet matching Gatys et al (2016)
    """
    
    if layers is None:
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1', 
                  '10': 'conv3_1', 
                  '19': 'conv4_1',
                  '21': 'conv4_2',  ## content representation
                  '28': 'conv5_1'}
        
        
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[layers[name]] = x
            
    return features

def _gram_matrix(tensor):
    """ Calculate the Gram Matrix of a given tensor 
        Gram Matrix: https://en.wikipedia.org/wiki/Gramian_matrix
    """
    
    _, depth, height, width = tensor.size()
    ## reshape it, so we're multiplying the features for each channel
    tensor = tensor.view(depth, height * width)
    ## calculate the gram matrix    
    gram = torch.mm(tensor, tensor.t())
    
    return gram

def _style_grams(style_features):
    return {
        layer: _gram_matrix(style_features[layer])
        for layer in style_features
    }

def _update_target_image(optimizer, total_loss):
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

def _content_loss(target_features, content_features):
    return torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)

def _layer_loss(target_gram, layer_gram, layer_weight, target_feature):
    _, depth, height, width = target_feature.shape
    return layer_weight * torch.mean((target_gram - layer_gram)**2) / (depth * height * width)

def _style_loss(style_weights, target_features, style_grams):
    style_loss = 0
    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = _gram_matrix(target_feature)
        layer_gram = style_grams[layer]
        layer_weight = style_weights[layer]
        style_loss += _layer_loss(layer_weight, target_gram, layer_gram)
    return style_loss

def _show_loss(total_loss, i=0, show_loss_every=1):
    loss = total_loss.item()
    if  i % show_loss_every == 0:
        print(f'step: {i}')
        print(f'loss: {loss}')
show_loss_every = 400

content_weight = 1  # alpha
style_weight = 1e6  # beta

vgg = _get_vgg19_features()
_freeze_vgg_parameters(vgg)
device = _try_use_gpu_as_torch_device()
vgg.to(device)
print(vgg)

content_path = 'https://vignette.wikia.nocookie.net/lovecraft/images/c/cf/Screenshot_20171018-093500.jpg'
content = _load_image(content_path).to(device)

style_path = 'https://d3d00swyhr67nd.cloudfront.net/w800h800/collection/SRY/RHU/SRY_RHU_THC0021-001.jpg'
style = _load_image(style_path, shape=content.shape[-2:]).to(device)

target = _get_copy_of_content_image(content, device)

content_features = _get_features(content, vgg)
style_features = _get_features(style, vgg)
style_grams = _style_grams(style_features)

# weighting earlier layers more will result in larger style artifacts
# `conv4_2` is excluded from the content style representation
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

learning_rate = 0.003
optimizer = optim.Adam(
    [target],
    lr=learning_rate,
)

steps = 5000
for i in range(steps):
    target_features = _get_features(target, vgg)
    content_loss = _content_loss(target_features, content_features)
    style_loss = _style_loss(style_weights, target_features, style_grams)
    total_loss = content_weight * content_loss + style_weight * style_loss

    _update_target_image(optimizer, total_loss)
    _show_loss(total_loss, i+1, show_loss_every)
_show_loss(total_loss)

initial_img = _img_from_np_array(_torch_tensor_to_np_image(content))
initial_img.save("initial_img.png")

final_img = _img_from_np_array(_torch_tensor_to_np_image(target))
final_img.save("final_img.png")