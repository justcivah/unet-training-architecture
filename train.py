from trainer.custom_segmentation_trainer import CustomSegmentationTrainer


args = dict(model="yolov8n-seg.pt", data="../tirocinio/datasets/scannet-dataset-small/dataset.yaml", epochs=1)
trainer = CustomSegmentationTrainer(overrides=args)
trainer.train()