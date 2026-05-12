"""SampConcatLowdimDataset — 合併多個 BaseLowdimDataset 供多任務訓練。

使用情境
--------
* 將 robomimic lift_ph + mimicgen_lift_d0 合併，擴增有效示範數量。
* 多個任務合併時，**obs_dim 與 action_dim 必須相同**（否則拋出 AssertionError）。

Config 用法（multi_gripper.yaml 中）：
    dataset:
      _target_: diffusion_policy.dataset.samp_concat_lowdim_dataset.SampConcatLowdimDataset
      datasets:
        - _target_: diffusion_policy.dataset.robomimic_replay_lowdim_dataset.RobomimicReplayLowdimDataset
          dataset_path: data/robomimic/datasets/lift/ph/low_dim_abs.hdf5
          ...
        - _target_: diffusion_policy.dataset.robomimic_replay_lowdim_dataset.RobomimicReplayLowdimDataset
          dataset_path: data/mimicgen/lift_d0/ph/low_dim.hdf5
          ...

注意事項
--------
* `get_normalizer()` 從所有子資料集的全部動作計算統一的 LinearNormalizer。
* `get_validation_dataset()` 回傳所有子資料集 val set 的合併版本。
* 每個子資料集各自維護自己的 train/val split（由各自的 val_ratio 控制）。
"""

from typing import List, Dict
import torch
import numpy as np

from diffusion_policy.dataset.base_dataset import BaseLowdimDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer


class SampConcatLowdimDataset(BaseLowdimDataset):
    """Concatenate multiple BaseLowdimDataset instances for multi-source training."""

    def __init__(self, datasets: List[BaseLowdimDataset]):
        super().__init__()
        assert len(datasets) >= 1, "Need at least one dataset"

        # ---- sanity check: all datasets must have the same shapes ----
        first_item = datasets[0][0]
        ref_obs_shape = first_item['obs'].shape
        ref_act_shape = first_item['action'].shape

        for i, ds in enumerate(datasets[1:], start=1):
            sample = ds[0]
            assert sample['obs'].shape == ref_obs_shape, (
                f"Dataset {i} obs shape {sample['obs'].shape} "
                f"!= dataset 0 obs shape {ref_obs_shape}"
            )
            assert sample['action'].shape == ref_act_shape, (
                f"Dataset {i} action shape {sample['action'].shape} "
                f"!= dataset 0 action shape {ref_act_shape}"
            )

        self.datasets = datasets
        # cumulative sizes for index mapping
        self._cumulative_sizes = torch.utils.data.ConcatDataset.cumsum(datasets)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self._cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # find which sub-dataset and local index
        dataset_idx = np.searchsorted(self._cumulative_sizes, idx, side='right')
        if dataset_idx == 0:
            local_idx = idx
        else:
            local_idx = idx - self._cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][local_idx]

    # ------------------------------------------------------------------
    def get_validation_dataset(self) -> 'SampConcatLowdimDataset':
        val_datasets = [ds.get_validation_dataset() for ds in self.datasets]
        return SampConcatLowdimDataset(val_datasets)

    def get_normalizer(self, mode='limits', **kwargs) -> LinearNormalizer:
        """Build a unified normalizer from all sub-datasets' actions and obs."""
        # collect all actions and obs from every sub-dataset
        all_obs = []
        all_actions = []
        for ds in self.datasets:
            for i in range(len(ds)):
                item = ds[i]
                all_obs.append(item['obs'])
                all_actions.append(item['action'])

        all_obs = torch.stack(all_obs)        # (N_total, T, Do)
        all_actions = torch.stack(all_actions)  # (N_total, T, Da)

        normalizer = LinearNormalizer()
        normalizer.fit(
            data={'obs': all_obs, 'action': all_actions},
            last_n_dims=1,
            mode=mode,
            **kwargs,
        )
        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        """Return all actions concatenated across sub-datasets."""
        return torch.cat([ds.get_all_actions() for ds in self.datasets], dim=0)

    # ---- informational helpers ----
    @property
    def n_episodes(self) -> int:
        return sum(getattr(ds, 'replay_buffer', type('', (), {'n_episodes': 0})()).n_episodes
                   for ds in self.datasets)

    def split_info(self) -> List[Dict]:
        """Return per-dataset split statistics."""
        info = []
        for i, ds in enumerate(self.datasets):
            n_train = len(ds)
            n_val = len(ds.get_validation_dataset())
            info.append({
                'dataset_idx': i,
                'n_train_samples': n_train,
                'n_val_samples': n_val,
                'class': type(ds).__name__,
            })
        return info
