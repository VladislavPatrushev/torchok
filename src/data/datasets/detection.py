import torch

from src.data import transforms as module_transforms
from torch.utils.data._utils.collate import default_collate

from src.registry import DATASETS
from .classification import ImageDataset


@DATASETS.register_class
class DetectionDataset(ImageDataset):
    """
    DetectionDataset class annotation_format:
    [x_min, y_min, x_max, y_max, label] - pascal_voc format in albumentation see the link
    https://albumentations.ai/docs/getting_started/bounding_boxes_augmentation/

    Example:
    [{'x_min': 520, 'y_min': 148, 'x_max': 600, 'y_max': 201, 'label': 20},
     {'x_min': 598, 'y_min': 206, 'x_max': 675, 'y_max': 240, 'label': 1}]

    :param target_column: Column name in csv file, with bboxes and labels in format wrote above
    :param min_area: Value in pixels. If the area of a bounding box after
     augmentation becomes smaller than min_area, Albumentations will drop that box
    :param min_visibility: Value between 0 and 1. If the ratio of the bounding box area after augmentation
     to the area of the bounding box before augmentation becomes smaller than min_visibility,
     Albumentations will drop that box.
    """

    def __init__(self,
                 target_column: str = 'annotation',
                 min_area: float = 0.0,
                 min_visibility: float = 0.0,
                 **dataset_params
                 ):
        super().__init__(**dataset_params)

        self.target_column = target_column
        if self.augment is not None:
            self.augment = module_transforms.Compose(
                self.augment,
                bbox_params=module_transforms.BboxParams(
                    format='pascal_voc',
                    label_fields=['category_ids'],
                    min_area=min_area,
                    min_visibility=min_visibility
                )
            )

        self.transform = module_transforms.Compose(
            self.transform,
            bbox_params=module_transforms.BboxParams(
                format='pascal_voc',
                label_fields=['category_ids'],
                min_area=min_area,
                min_visibility=min_visibility
            )
        )

        self.csv[target_column] = self.csv[target_column].apply(eval)

    def __getitem__(self, idx: int):
        sample = self.get_raw(idx // self.expand_rate)
        sample['image'] = sample['image'].type(torch.__dict__[self.input_dtype])

        output = {
            'input': sample['image'],
            'target_bboxes': torch.tensor(sample['bboxes']).type(torch.__dict__[self.target_dtype]),
            'target_classes': torch.tensor(sample['category_ids']).type(torch.long),
            'bbox_count': torch.tensor(sample['bbox_count']),
            'pad_shape': torch.tensor(sample['pad_shape'], dtype=torch.long),
            'img_shape': torch.tensor(sample['img_shape'], dtype=torch.long),
            'scale_factor': torch.tensor(sample['scale_factor'], dtype=torch.__dict__[self.target_dtype])
        }

        return output

    def get_raw(self, idx: int):
        record = self.csv.iloc[idx]
        image = self.read_image(record)
        orig_shape = image.shape[:2]
        row_annotations = record[self.target_column]

        bboxes = []
        classes = []
        for annotation in row_annotations:
            bbox = [int(annotation['x_min']), int(annotation['y_min']), int(annotation['x_max']),
                    int(annotation['y_max'])]
            label = annotation['label']
            bboxes.append(bbox)
            classes.append(label)

        sample = {
            'image': image,
            'bboxes': bboxes,
            'category_ids': classes
        }

        if self.augment is not None:
            sample = self.augment(**sample)

        sample = self.transform(**sample)
        sample['bbox_count'] = len(sample['bboxes'])
        sample['pad_shape'] = sample['image'].shape[1:3]
        sample['img_shape'] = orig_shape
        # TODO: define scale_factor
        sample['scale_factor'] = (1.0, 1.0, 1.0, 1.0)

        return sample

    def collate_fn(self, batch: dict) -> dict:
        """
        Pad bboxes and labels tensors with empty data to form a fix shaped output tensors.
        Size of the corresponding dimension is equal to the maximum number of bboxes in the given batch.
        empty bbox = [0, 0, 0, 0]
        empty label = -1
        """
        # get maximum sequence length
        max_length = 0
        for t in batch:
            max_length = max(max_length, t['bbox_count'])

        if max_length != 0:
            for t in batch:
                bboxes = torch.zeros(max_length, 4, dtype=torch.__dict__[self.target_dtype])
                labels = torch.full((max_length,), -1, dtype=torch.long)
                bbox_count = t['bbox_count']
                if bbox_count != 0:
                    bboxes[:bbox_count] = t['target_bboxes']
                    labels[:bbox_count] = t['target_classes']
                t['target_bboxes'] = bboxes
                t['target_classes'] = labels

        batch = default_collate(batch)

        return batch