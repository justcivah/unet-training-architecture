from models.yolo.model import SegmentationModel

class CombinedModel(SegmentationModel): 
    def __init__(self, unet, yolo):
        super().__init__()
        self.unet = unet
        self.yolo = yolo
        
        # Lock YOLO weights
        for param in self.yolo.parameters():
            param.requires_grad = False
            
        # Set model parameters for trainer compatibility
        self.args = self.yolo.args
        self.names = self.yolo.names
        self.stride = self.yolo.stride
        self.task = 'segment'

    def forward(self, batch):
        # Pass through UNet first
        unet_out = self.unet(batch['img']) 
        
        # Replace image with UNet output in batch
        batch['img'] = unet_out
        
        # Pass through YOLO and get loss
        loss = self.yolo(batch)
        return loss

    def predict(self, batch, **kwargs):
        unet_out = self.unet(batch['img'])
        batch['img'] = unet_out
        return self.yolo.predict(batch, **kwargs)