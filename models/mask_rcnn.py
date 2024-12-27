import torch
import os
import warnings
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.checkpoint import DetectionCheckpointer

from detectron2.data import build_detection_test_loader  # Import for test function

from configs.dataset_mapping import register_datasets
from detectron2.utils.logger import setup_logger
from detectron2.data import DatasetCatalog, build_detection_train_loader






class MaskRCNNModel:
    def __init__(self, config_path):
        self.config_path = config_path
        self.cfg = None
        self.trainer = None
        


    def _validate_dataset(self, dataset_name):

        if dataset_name not in DatasetCatalog.list():
            raise ValueError(f"Dataset '{dataset_name}' is not registered.")
        dataset = DatasetCatalog.get(dataset_name)

        if not dataset:
            raise ValueError(f"Dataset '{dataset_name}' is empty or not loaded correctly.")
        print(f"Dataset '{dataset_name}' has {len(dataset)} samples.")


    def train(self, device=None, output_dir="./result"):

         # Suppress warnings for torch.meshgrid deprecation
        warnings.filterwarnings("ignore", message="torch.meshgrid: in an upcoming release, it will be required to pass the indexing argument.")

        # Setup Detectron2 logger
        logger =setup_logger(output=output_dir, name="detectron2")
        logger.setLevel("DEBUG")

        cfg = get_cfg()
        cfg.merge_from_file(self.config_path)

        # Set the training device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Set the training device
        cfg.MODEL.DEVICE = device

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        cfg.OUTPUT_DIR = output_dir


        # Validate datasets
        self._validate_dataset(cfg.DATASETS.TRAIN[0])  # Check training dataset
        self._validate_dataset(cfg.DATASETS.TEST[0])   # Check validation dataset

        # # Ignore unused keys warnings in pretrained weights
        # cfg.MODEL.WEIGHTS_IGNORE_MISSING_KEYS = True

        # Log important training parameters
        logger.info(f"Starting training on device: {device}")
        logger.info(f"Configuration file: {self.config_path}")
        logger.info(f"Output directory: {output_dir}")

        trainer = DefaultTrainer(cfg)
        trainer.resume_or_load(resume=False)

        if cfg.MODEL.WEIGHTS:
            logger.info(f"Loading weights from: {cfg.MODEL.WEIGHTS}")
            checkpointer = DetectionCheckpointer(trainer.model)
            checkpointer.load(cfg.MODEL.WEIGHTS)
            

        # Add custom hooks for debugging
        from detectron2.engine import hooks

        class DebugHook(hooks.HookBase):
            def after_step(self):
                print(f"Iteration {self.trainer.iter} completed.")

        trainer.register_hooks([DebugHook()])

        try:
            trainer.train()
            logger.info("Training Successful")
        except Exception as e:
            logger.error("Training Failed : {e}")
    
    def test(self, dataset_name):
        # Ensure the configuration and trainer are initialized
        if not self.cfg or not self.trainer:
            raise ValueError("You must train the model before testing.")

        self.cfg.MODEL.WEIGHTS = "./result/checkpoints/"  # Ensure this path is correct

        evaluator = COCOEvaluator(dataset_name, self.cfg, False, output_dir="./result/predictions/")
        val_loader = build_detection_test_loader(self.cfg, dataset_name)
        
         # Run inference on the dataset
        return inference_on_dataset(self.trainer.model, val_loader, evaluator)