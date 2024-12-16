from ultralytics.engine.model import Model

class CombinedModel(Model):
    def __init__(self, unet, yolo): pass