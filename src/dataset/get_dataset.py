import torch
import numpy as np
from torch.utils.data import Dataset
import pdb


class DataFrameDataset(Dataset):
    def __init__(self, dataframe, feature_cols, target_col, local_cols, transform=None):
        """
        Args:
            dataframe (pd.DataFrame): Input data.
            feature_cols (list): List of column names to be used as features.
            target_col (str): Name of the target column.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dataframe = dataframe
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.local_cols = local_cols
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # breakpoint()
        features = self.dataframe.iloc[idx][self.feature_cols].to_numpy(dtype=np.float32)
        target = self.dataframe.iloc[idx][self.target_col]
        sample = {'features': torch.from_numpy(features),
                  'target': torch.tensor(target, dtype=torch.float32)}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def get_features(self):
        """
        Returns:
            List of feature column names.
        """
        return self.feature_cols
    
    def get_local_cols(self):
        """
        Returns:
            List of local column names.
        """
        return self.local_cols
    
    def get_target(self):
        """
        Returns:
            Name of the target column.
        """
        return self.target_col


def get_dataloader(dataset,
                    batch_size=32, shuffle=True, transform=None, one_batch=False):
    """
    Create a DataLoader for the given Dataset.
    Args:
        dateset (DataFrameDataset): The dataset to load data from.
        batch_size (int): Size of each batch.
        shuffle (bool): Whether to shuffle the data.
        transform (callable, optional): Optional transform to be applied on a sample.
    Returns:
        DataLoader: DataLoader for the dataset.
    """
    if one_batch:
        batch_size = dataset.__len__()
  
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

