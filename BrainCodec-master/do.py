from audiocraft.solvers.compression_fMRI import CompressionfMRISolver
import logging
from torchinfo import summary
import torch

import torch.nn as nn
logging.basicConfig(level=logging.INFO)
import os
# Ensure USER variable exists
if 'USER' not in os.environ:
    os.environ['USER'] = 'default_username' 
path_to_checkpoint = "E:/recherche/brain/brain_codec/BrainCodec-master/BrainCodec-master/checkpoint_200.th"
modelencoder = CompressionfMRISolver.model_from_checkpoint(path_to_checkpoint, device="cpu")
#print("the out put is = ",modelencoder)
print(" be happy the model is loaded successfully Ps the size of the model is  ", type(modelencoder))
# it has just to be a 3 D  dim 
variable = 32
random_tensor = torch.randn( 450,  1024, variable)  # Taille adaptée à ton modèle #la taille doit avoir la forme ( , 1024, )
tr = torch.zeros(450, variable)  # Example additional tensor if required
tr = tr.long()
tr=tr.to('cpu')
# Passage avant (forward pass)
result = modelencoder(random_tensor,tr)



# Assuming `result` is the output from the model
print("Shape of x (quantized result):", result.x.shape)  # Shape of x
print("Shape of codes:", result.codes.shape)            # Shape of codes
print("Shape of bandwidth (scalar):", result.bandwidth.shape)  # Bandwidth is likely a scalar
print("Shape of penalty (scalar):", result.penalty.shape)      # Penalty is likely a scalar

# If metrics is a dictionary, print the keys and shapes of its contents
if isinstance(result.metrics, dict):
    print("Metrics keys:", result.metrics.keys())
    for key, value in result.metrics.items():
        print(f"Shape of metrics['{key}']:", value.shape if isinstance(value, torch.Tensor) else "Not a Tensor")


"""encoder = modelencoder.encoder 
print("Encoder Summary:")
summary(encoder, input_data=torch.randn(450, 1024, 256))
breakpoint()"""

"""
print( 'okkkkkkkkkkkkk le s gooooooooooo')
class SummaryWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        q_res = self.model(*args, **kwargs)
        if hasattr(q_res, 'x'):  # Check if q_res has an 'x' attribute
            return q_res.x
        return q_res # Only return tensor for summary

wrapped_model = SummaryWrapper(modelencoder)
summary(wrapped_model, input_data=torch.randn(450, 1024, 32))

breakpoint()"""
#print('the out put of the model is: {}'.format(result))
print( 'hey hey the summary is here ')
breakpoint()
summary(modelencoder, input_data=torch.randn(450, 1024, variable))
