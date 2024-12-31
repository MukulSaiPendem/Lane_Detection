import torch
import torch.nn as nn
import torchvision.models as models
from .attention import CoordAttention

class LaneDetectionModel(nn.Module):
    """
    Lane Detection Model with ResNet backbone (18 or 50) and Coordinate Attention
    """
    def __init__(self, 
                 num_classes: int = 2,
                 backbone: str = 'resnet50',
                 pretrained: bool = True):
        """
        Initialize the Lane Detection Model.
        
        Args:
            num_classes (int): Number of output classes
            backbone (str): Backbone model name ('resnet18' or 'resnet50')
            pretrained (bool): Whether to use pretrained backbone
        """
        super(LaneDetectionModel, self).__init__()
        
        # Initialize backbone
        self.backbone_name = backbone
        if backbone == 'resnet50':
            resnet = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            encoder_channels = 2048
        elif backbone == 'resnet18':
            resnet = models.resnet18(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(resnet.children())[:-2])
            encoder_channels = 512
        else:
            raise ValueError(f"Unsupported backbone: {backbone}. Choose 'resnet18' or 'resnet50'")
        
        # Initialize attention
        self.coord_att = CoordAttention(encoder_channels, encoder_channels)
        
        # Decoder layers - Adapted for both ResNet18 and ResNet50
        if backbone == 'resnet50':
            self.decoder = nn.ModuleList([
                # Up1: 2048 -> 1024 -> 512
                nn.Sequential(
                    nn.Conv2d(encoder_channels, 1024, kernel_size=3, padding=1),
                    nn.BatchNorm2d(1024),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
                ),
                # Up2: 512 -> 256 -> 128
                nn.Sequential(
                    nn.Conv2d(512, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                ),
                # Up3: 128 -> 64 -> 32
                nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
                )
            ])
        else:  # ResNet18
            self.decoder = nn.ModuleList([
                # Up1: 512 -> 256 -> 128
                nn.Sequential(
                    nn.Conv2d(encoder_channels, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
                ),
                # Up2: 128 -> 64 -> 32
                nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
                ),
                # Up3: 32 -> 32 -> 32
                nn.Sequential(
                    nn.Conv2d(32, 32, kernel_size=3, padding=1),
                    nn.BatchNorm2d(32),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
                )
            ])
        
        # Final classification layer
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, kernel_size=1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Output segmentation map
        """
        # Encoder
        features = self.encoder(x)
        
        # Apply attention
        features = self.coord_att(features)
        
        # Decoder
        x = features
        for decoder_block in self.decoder:
            x = decoder_block(x)
        
        # Final prediction
        x = self.final_conv(x)
        
        return x

    def get_backbone_parameters(self):
        """Get backbone parameters for separate optimization"""
        return self.encoder.parameters()

    def get_decoder_parameters(self):
        """Get decoder parameters for separate optimization"""
        params = []
        params.extend(self.decoder.parameters())
        params.extend(self.final_conv.parameters())
        return params

    def print_model_info(self):
        """Print model architecture information"""
        print(f"\nModel Architecture Info:")
        print(f"Backbone: {self.backbone_name}")
        print(f"Number of parameters:")
        print(f"- Encoder: {sum(p.numel() for p in self.encoder.parameters())}")
        print(f"- Attention: {sum(p.numel() for p in self.coord_att.parameters())}")
        print(f"- Decoder: {sum(p.numel() for p in self.decoder.parameters())}")
        print(f"- Final Conv: {sum(p.numel() for p in self.final_conv.parameters())}")
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Total Parameters: {total_params}")