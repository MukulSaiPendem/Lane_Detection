import torch
import torch.nn as nn

class CoordAttention(nn.Module):
    """
    Coordinate Attention Mechanism for spatial and channel focus.
    """
    def __init__(self, in_channels: int, out_channels: int, reduction: int = 32):
        """
        Initialize the Coordinate Attention module.
        
        Args:
            in_channels (int): Number of input channels
            out_channels (int): Number of output channels
            reduction (int): Channel reduction factor
        """
        super(CoordAttention, self).__init__()
        
        # Pooling layers
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        # Calculate mid channels
        mid_channels = max(8, in_channels // reduction)
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv_h = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
        self.conv_w = nn.Conv2d(mid_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the attention module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Attention-modulated output tensor
        """
        identity = x
        n, c, h, w = x.size()
        
        # Spatial pooling
        x_h = self.pool_h(x)
        x_w = self.pool_w(x)
        
        # Channel reduction
        x_h = self.conv1(x_h)
        x_w = self.conv1(x_w)
        
        # Spatial attention
        x_h = x_h.expand(-1, -1, h, w)
        x_w = x_w.expand(-1, -1, h, w)
        
        # Combine features
        combined = x_h + x_w
        combined = self.relu(combined)
        
        # Generate attention maps
        att_h = self.conv_h(combined)
        att_w = self.conv_w(combined)
        
        # Apply attention
        attention = torch.sigmoid(att_h + att_w)
        
        return attention * identity