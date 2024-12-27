import cv2
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from configs.dataset_mapping import register_datasets

if __name__ == "__main__":
    register_datasets()

    cfg = get_cfg()
    cfg.merge_from_file("./configs/mask_rcnn_config.yaml")
    predictor = DefaultPredictor(cfg)

    image_path = "./datasets/datasets_a/images/example.jpg"
    image = cv2.imread(image_path)
    outputs = predictor(image)

    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]))
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    cv2.imshow("Inference", out.get_image()[:, :, ::-1])
    cv2.waitKey(0)