import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
import numpy as np

class PrepareDataset(torch.utils.data.Dataset):
    def __init__(self, df, scaler, mode="a", include_is_dist = False):
        self.mode = mode
        self.scaler = scaler
        self.include_is_dist = include_is_dist # is_dist is used for the disturbances subset
        if include_is_dist:
            self.is_dist = torch.tensor(df["is_dist"].values, dtype=torch.bool)

        self.lon = torch.tensor(df["lon"].values, dtype=torch.float32)
        self.lat = torch.tensor(df["lat"].values, dtype=torch.float32)

        def _scale_a(a_phys):
            if hasattr(self.scaler, "scale_acceleration"):
                return self.scaler.scale_acceleration(a_phys)
            raise AttributeError(
                "Scaler must implement scale_acceleration."
            )

        if mode == "u":
            u_phys = df["dU_m2_s2"].values
            u_scaled = self.scaler.scale_potential(u_phys)
            self.y = torch.tensor(u_scaled, dtype=torch.float32).unsqueeze(1)

        else:
            cols = ["dg_theta_mGal", "dg_phi_mGal", "dg_r_mGal"]
            a_phys = df[cols].values
            a_scaled = _scale_a(a_phys)
            self.y = torch.tensor(a_scaled, dtype=torch.float32)

    def __len__(self):
        return len(self.lon)

    def __getitem__(self, idx):
        if self.include_is_dist:
            return self.lon[idx], self.lat[idx], self.y[idx], self.is_dist[idx]
        return self.lon[idx], self.lat[idx], self.y[idx]


class SplitDataset(pl.LightningDataModule):
    def __init__(self, train_df, val_df, test_df, scaler, mode="a", batch_size=10240):
        super().__init__()
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.scaler = scaler
        self.mode = mode
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.train_dataset = PrepareDataset(self.train_df, self.scaler, mode=self.mode)
        self.val_dataset = PrepareDataset(self.val_df, self.scaler, mode=self.mode)

        # compute disturbance mask only for test
        A = self.test_df.copy()
        g = A["dg_total_mGal"].values
        mean_g = g.mean()
        std_g = g.std()
        A["is_dist"] = (np.abs(g - mean_g) > 2 * std_g)

        self.test_dataset = PrepareDataset(A, self.scaler, mode=self.mode, include_is_dist=True)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=2,
                          pin_memory=torch.cuda.is_available(), persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=2,
                          pin_memory=torch.cuda.is_available(), persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size,
                          shuffle=False, num_workers=2,
                          pin_memory=torch.cuda.is_available(), persistent_workers=True)
