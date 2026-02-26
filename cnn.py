from __init__ import *
    # Input: [N, (1), 49 * 3, 49 * 3]; Output: [N, 2]
    # Two Convolution Blocks
class CNN49u(nn.Module):
  def __init__(self):
    super().__init__()
    # * Below build the structure of neural network with order Conv2d -> BN -> ReLU(Activate Function) -> Max-Pool
    self.conv1 = nn.Sequential(OrderedDict([
      ('Conv', nn.Conv2d(1, 64, (5, 3), padding = (2, 1), stride = (1, 1), dilation = (1, 1))), 
      # Grayscale: no. in-Channel is 1
      # Filter/kernel size (5, 3), e.g. flat, small jump, big jump in the reference
      # No stride, and no dilation
      # Manual chose no. out-channel as 64
      # Manual chose padding size (2, 1) to ensure the output size of each image matrix is still 147 * 147
      # Output size: [N, 64, 147, 147]
      ('BN', nn.BatchNorm2d(64, affine = True)),
      # Output size: [N, 64, 147, 147]
      ('ReLU', nn.ReLU()),
      # ? Could also consider leaky ReLU, i.e. nn.LeakyReLU(negative_slope = 5e-2), but note the slope for negative inputs need specifying
      # Output size: [N, 64, 147, 147]
      ('Max-Pool', nn.MaxPool2d(kernel_size = 2, ceil_mode = True))
      # Extract the dominant feature within the window
      # Could also consider max-pooling window 3 * 3, but may lose too much data
      # Since 147 is divisible by 2, consider ceil_mode
      # Output size: [N, 64, 74, 74]
    ]))
    self.conv1 = self.conv1.apply(self.init_weights)
    # Initialize the weights of the layer by Xaxier Uniform Initialization, which keeps scale of gradient roughly the same and avoid
    # gradient vanishing
    
    self.conv2 = nn.Sequential(OrderedDict([
      ('Conv', nn.conv2d(64, 128, (5, 3), padding = (2, 1), stride = (1, 1), dilation = (1, 1))),
      # Output: [N, 128, 74, 74]
      ('BN', nn.BatchNorm2d(128, affine = True)),
      # Output: [N, 128, 74, 74]
      ('ReLU', nn.ReLU()),
      # Output: [N, 128, 74, 74]
      ('Max-Pool', nn.MaxPool2d((2, 2)))
      # Output: [N, 128, 37, 37] 
    ]))
    self.conv2 = self.conv2.apply(self.init_weights)
    
    self.DropOut = nn.Dropout(p = 0.5)
    # Zero the activations by probability 0.5 to regularize the dense connections
    self.FC = nn.Linear(128 * 37 * 37, 2)
    # Flatten the data to 2-dimension
    self.init_weights(self.FC)
    self.Softmax = nn.Softmax(dim = 1)
    # Output the label for classification from the data by the model
    
  def init_weights(self, model):
      if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.01)

  def forward(self, x):
    # x: input [N, 147, 147]
    x = x.unsqueeze(1).to(torch.float32)
    # Now x lacks a dimension of channel. In grayscale, it is 1. 
    # unsqueeze(1) inserts a dimension of data at the position 1 to align with the
    # channel input format
    # float32 converts the current tensor object to 32-bit tensor for compatibility
    # since PyTorch operations sometimes require 32-bit input.
    # Output [N, 1, 147, 147]
    x = self.conv1(x)
    # Output [N, 64, 74, 74]
    x = self.conv2(x)
    # Output [N, 128, 37, 37]
    x = self.DropOut(x.view(x.shape[0], -1))
    x = self.FC(x)
    # Output [N, 2]
    x = self.Softmax(x)
    # Output [N, 1]
    return x
  
class CNN9u(nn.Module):
  def __init__(self):
    super(CNN9u, self).__init__()
    # * Below build the structure of neural network with order Conv2d -> BN -> ReLU(Activate Function) -> Max-Pool
    self.conv1 = nn.Sequential(OrderedDict([
      ('Conv', nn.Conv2d(1, 64, (5, 3), padding = (2, 1), stride = (1, 1), dilation = (1, 1))), 
      # Grayscale: no. in-Channel is 1
      # Filter/kernel size (5, 3), e.g. flat, small jump, big jump in the reference
      # No stride, and no dilation
      # Manual chose no. out-channel as 64
      # Manual chose padding size (2, 1) to ensure the output size of each image matrix is still 27 * 27
      # Output size: [N, 64, 27, 27]
      ('BN', nn.BatchNorm2d(64, affine = True)),
      # Output size: [N, 64, 27, 27]
      ('ReLU', nn.ReLU()),
      # ? Could also consider leaky ReLU, i.e. nn.LeakyReLU(negative_slope = 5e-2), but note the slope for negative inputs need specifying
      # Output size: [N, 64, 27, 27]
      ('Max-Pool', nn.MaxPool2d(kernel_size = 2, ceil_mode = True))
      # Extract the dominant feature within the window
      # Could also consider max-pooling window 3 * 3, but may lose too much data
      # Since 27 is divisible by 2, consider ceil_mode
      # Output size: [N, 64, 14, 14]
    ]))
    self.conv1 = self.conv1.apply(self.init_weights)
    # Initialize the weights of the layer by Xaxier Uniform Initialization, which keeps scale of gradient roughly the same and avoid
    # gradient vanishing
    
    self.conv2 = nn.Sequential(OrderedDict([
      ('Conv', nn.conv2d(64, 128, (5, 3), padding = (2, 1), stride = (1, 1), dilation = (1, 1))),
      # Output: [N, 128, 14, 14]
      ('BN', nn.BatchNorm2d(128, affine = True)),
      # Output: [N, 128, 14, 14]
      ('ReLU', nn.ReLU()),
      # Output: [N, 128, 14, 14]
      ('Max-Pool', nn.MaxPool2d((2, 2)))
      # Output: [N, 128, 7, 7] 
    ]))
    self.conv2 = self.conv2.apply(self.init_weights)
    
    self.DropOut = nn.Dropout(p = 0.5)
    # Zero the activations by probability 0.5 to regularize the dense connections
    self.FC = nn.Linear(128 * 7 * 7, 2)
    # Flatten the data to 2-dimension
    self.init_weights(self.FC)
    self.Softmax = nn.Softmax(dim = 1)
    # Output the label for classification from the data by the model
    
  def init_weights(self, model):
      if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
        torch.nn.init.xavier_uniform(model.weight)
        model.bias.data.fill_(0.01)

  def forward(self, x):
    # x: input [N, 27, 27]
    x = x.unsqueeze(1).to(torch.float32)
    # Now x lacks a dimension of channel. In grayscale, it is 1. 
    # unsqueeze(1) inserts a dimension of data at the position 1 to align with the
    # channel input format
    # float32 converts the current tensor object to 32-bit tensor for compatibility
    # since PyTorch operations sometimes require 32-bit input.
    # Output [N, 1, 27, 27]
    x = self.conv1(x)
    # Output [N, 64, 14, 14]
    x = self.conv2(x)
    # Output [N, 128, 14, 14]
    x = self.DropOut(x.view(x.shape[0], -1))
    x = self.FC(x)
    # Output [N, 2]
    x = self.Softmax(x)
    # Output [N, 1]
    return x