from detectron2.data import DatasetCatalog, MetadataCatalog

# Utility functions for dataset management
def get_metadata(dataset_name):
    return MetadataCatalog.get(dataset_name)

def get_dataset_dicts(dataset_name):
    return DatasetCatalog.get(dataset_name)