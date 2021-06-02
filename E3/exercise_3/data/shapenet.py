from pathlib import Path
import json

import numpy as np
import torch


class ShapeNet(torch.utils.data.Dataset):
    num_classes = 8
    dataset_sdf_path = Path("exercise_3/data/shapenet_dim32_sdf")  # path to voxel data
    dataset_df_path = Path("exercise_3/data/shapenet_dim32_df")  # path to voxel data
    class_name_mapping = json.loads(Path("exercise_3/data/shape_info.json").read_text())  # mapping for ShapeNet ids -> names
    classes = sorted(class_name_mapping.keys())

    def __init__(self, split):
        super().__init__()
        assert split in ['train', 'val', 'overfit']
        self.truncation_distance = 3

        self.items = Path(f"exercise_3/data/splits/shapenet/{split}.txt").read_text().splitlines()  # keep track of shapes based on split

    def __getitem__(self, index):
        sdf_id, df_id = self.items[index].split(' ')

        input_sdf = ShapeNet.get_shape_sdf(sdf_id)
        target_df = ShapeNet.get_shape_df(df_id)

        input_sdf = np.clip(input_sdf, a_min=-self.truncation_distance, a_max=self.truncation_distance)
        input_sdf = np.stack([np.abs(input_sdf), np.sign(input_sdf)])  # (distances, sdf sign)
        target_df = np.clip(target_df, a_min=0, a_max=self.truncation_distance)
        target_df = np.log(np.add(target_df, 1.0))

        return {
            'name': f'{sdf_id}-{df_id}',
            'input_sdf': input_sdf,
            'target_df': target_df
        }

    def __len__(self):
        return len(self.items)

    @staticmethod
    def move_batch_to_device(batch, device):
        # TODO add code to move batch to device
        batch['name'] = batch['name'].to(device)
        batch['input_sdf'] = batch['input_sdf'].to(device)
        batch['target_df'] = batch['target_df'].to(device)

    @staticmethod
    def get_shape_sdf(shapenet_id):
        sdf = None
        # TODO implement sdf data loading
        with open(ShapeNet.dataset_sdf_path / (shapenet_id + '.sdf'),"rb") as f:
            #sdf = np.fromfile(f, dtype=np.dtype([('dims',np.uint64,3),('values',np.float32,32*32*32)]))
            bytes = f.read()
            dimX = int.from_bytes(bytes[:8],'little')
            dimY = int.from_bytes(bytes[8:16],'little')
            dimZ = int.from_bytes(bytes[16:24],'little')
            
            sdf = np.frombuffer(bytes[24:],dtype=np.float32)
            sdf = sdf.reshape(dimX,dimY,dimZ)
            
        return sdf

    @staticmethod
    def get_shape_df(shapenet_id):
        df = None
        # TODO implement df data loading
        with open(ShapeNet.dataset_df_path / (shapenet_id + '.df'),"rb") as f:
            #df = np.fromfile(f, dtype=np.dtype([('dims',np.uint64,3),('values',np.float32,32*32*32)]))
            bytes = f.read()
            dimX = int.from_bytes(bytes[:8],'little')
            dimY = int.from_bytes(bytes[8:16],'little')
            dimZ = int.from_bytes(bytes[16:24],'little')
            
            df = np.frombuffer(bytes[24:],dtype=np.float32)
            df = df.reshape(dimX,dimY,dimZ)

        return df
