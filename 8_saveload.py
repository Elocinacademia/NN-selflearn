# Save and Load the Model

import torch
import torchvision.models as models


# Save and loading Model Weights
'''
PyTorch models store the learned parameters in an internal state dictionary, called `state_dict`. These can be persisted via the `torch.save` method.
'''

model = models.vgg16(pretrained = True)
torch.save(model.state_dict(), 'model_weights.pth')


'''
To load model weights, you need to create an instance of the same model first, and then load the parameters using `load_state_dict()` method.
'''

model = models.vgg16()  ## We donnot specify pretrained = True, i.e. do not load the default weights
model.load_state_dict(torch.load('model_weights.pth'))
model.eval()

'''
Note: be sure to call model.eval() method before inferencing to set the dropout and batch normalization layers to evaluation mode. Failing to do this will yield inconsistent inference results.
'''

# Saving and Loading Models with Shapes
'''
When loading model weights, we needed to instantiate the model class first, because the class defines the structure of a network. We might want to save the structure of this class together with the model, in which case we can pass `model` (and not `model.state_dict()`) to the saving function:
'''

torch.save(model, 'model.pth')

'''
Then we can load the model like this:
'''

model = torch.load('model.pth')

'''
Note: This approach uses Python pickle module when serializing the model, thus it relies on the actual class definition to be available when loading the model.
'''


