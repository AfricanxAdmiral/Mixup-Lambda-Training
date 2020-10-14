import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.nn.parameter import Parameter

def init_layer(L):
  # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0]*L.kernel_size[1]*L.out_channels
        L.weight.data.normal_(0,math.sqrt(2.0/float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)

# --- Conv2d module ---
class Conv2d_fw(nn.Conv2d): #used in MAML to forward input with fast weight
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=0, bias = True):
        super(Conv2d_fw, self).__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias)
        # print("use fw")
        self.weight.fast = None
        if not self.bias is None:
            self.bias.fast = None

    def forward(self, x):
        gb = False
        if self.bias is None:
            if self.weight.fast is not None:                             
                out = F.conv2d(x, self.weight.fast, None, stride= self.stride, padding=self.padding)
                # print("use fast")
            else:                            
                out = super(Conv2d_fw, self).forward(x)
                # print("no")
        else:
            if self.weight.fast is not None and self.bias.fast is not None:                
                out = F.conv2d(x, self.weight.fast, self.bias.fast, stride= self.stride, padding=self.padding)
                # print("use fast")
            else:                
                out = super(Conv2d_fw, self).forward(x)
                # print("no")
        return out   


# --- BatchNorm2d ---
class BatchNorm2d_fw(nn.BatchNorm2d):
    def __init__(self, num_features, momentum=0.1, track_running_stats=True):
        super(BatchNorm2d_fw, self).__init__(num_features, momentum=momentum, track_running_stats=track_running_stats) 
        self.label = None
        
        self.weight.fast = None
        self.bias.fast = None

        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.zeros(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            
    def forward(self, x, label=None, step=0):
        if self.weight.fast is not None and self.bias.fast is not None:
            weight = self.weight.fast
            bias = self.bias.fast
            # print("use fast")
        else:
            weight = self.weight
            bias = self.bias
            # print("no")
        
        if self.track_running_stats:
            out = F.batch_norm(x, self.running_mean, self.running_var, weight, bias, training=self.training, momentum=self.momentum)
        else:
            out = F.batch_norm(x, torch.zeros(x.size(1), dtype=x.dtype, device=x.device), torch.ones(x.size(1), dtype=x.dtype, device=x.device), weight, bias, training=True, momentum=1)
        return out
