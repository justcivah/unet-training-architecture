from trainer.segmentation_trainer import SegmentationTrainer


args = dict(model="yolov8n-seg.pt", data="scannet-dataset-test/dataset.yaml", epochs=1)
trainer = SegmentationTrainer(overrides=args)
trainer.train()