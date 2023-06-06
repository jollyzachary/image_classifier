from imports import *

#IMAGE PROCESSING
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array and a PIL image
    '''
    # Open the image
    im = Image.open(image)
    
    # Resize the image so the shortest side is 256 pixels
    im.thumbnail((256, 256))
    
    # Crop out the center 224x224 portion of the image
    width, height = im.size
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    im = im.crop((left, top, right, bottom))
    
    # Convert the image to a numpy array and normalize the pixel values
    np_image = np.array(im) / 255
    
    # Standardize the pixel values
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std
    
    # Transpose the color channel to be the first dimension
    np_image = np_image.transpose((2, 0, 1))
    
    # Convert the numpy array to a tensor
    tensor_image = torch.from_numpy(np_image).type(torch.FloatTensor)
    
    return tensor_image, im

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax

def display_image_predictions(image, probs, flowers):
    ''' Function for viewing an image and it's predicted classes.
    '''
    fig, (ax1, ax2) = plt.subplots(figsize=(6,10), ncols=1, nrows=2)
    ax1.imshow(image)
    ax1.axis('off')
    
    y_pos = np.arange(len(flowers))
    ax2.barh(y_pos, probs)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(flowers)
    ax2.invert_yaxis()  # probabilities read top-to-bottom
    ax2.set_title('Class Probability')
    plt.tight_layout()
    plt.show()

