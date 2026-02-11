# Copyright 2022 Cerebras Systems.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pytorch HuggingFace OLMo Mix Iterable Dataloader."""

# pyright: reportAttributeAccessIssue=false

from typing import Any, Literal, Optional, Sequence, cast

from pydantic import Field

import torch
from datasets.distributed import split_dataset_by_node

from cerebras.modelzoo.common.input_utils import get_streaming_batch_size
from cerebras.modelzoo.config import DataConfig
from cerebras.modelzoo.data.common.input_utils import num_tasks, task_id
from cerebras.modelzoo.data_preparation.huggingface.HuggingFace_OlmoMix import (
    HuggingFace_OlmoMix,
)


class HuggingFaceIterableDataProcessorOlmoMixConfig(DataConfig):
    data_processor: Literal["HuggingFaceIterableDataProcessorOlmoMix"]

    batch_size: int = 1
    shuffle: bool = False
    shuffle_seed: Optional[int] = None
    shuffle_buffer: Optional[int] = None
    num_workers: int = 0
    drop_last: bool = True
    prefetch_factor: Optional[int] = 10
    persistent_workers: bool = True

    split: str = "train"
    subset: str = "default"
    sequence_length: int = 2048
    tokenizer_name: str = "gpt2"
    max_samples: Optional[int] = Field(10000, ge=1)
    streaming: bool = True
    use_data_files: bool = True
    max_files: Optional[int] = Field(2, ge=1)
    data_files: Optional[Any] = None
    cache_dir: Optional[str] = None
    use_local_data: bool = False
    data_root: Optional[str] = None
    subsets: Sequence[str] | str = "wiki"
    files_per_subset: Optional[int] = Field(2, ge=1)
    use_weighted_files: bool = False
    weighted_file_list: Optional[str] = None
    sampling_seed: int = 0
    loop_files: bool = True
    debug_log_first_n: int = 0
    debug_text_words: int = 10


class HuggingFaceIterableDataProcessorOlmoMix:
    """
    A HuggingFace OLMo Mix Iterable Data Processor.

    Args:
        config: The configuration object
    """

    def __init__(self, config: HuggingFaceIterableDataProcessorOlmoMixConfig):
        if isinstance(config, dict):
            config = HuggingFaceIterableDataProcessorOlmoMixConfig(**config)

        self.dataset, self.data_collator = HuggingFace_OlmoMix(
            split=config.split,
            subset=config.subset,
            num_workers=config.num_workers,
            sequence_length=config.sequence_length,
            tokenizer_name=config.tokenizer_name,
            max_samples=config.max_samples,
            streaming=config.streaming,
            use_data_files=config.use_data_files,
            max_files=config.max_files,
            data_files=config.data_files,
            cache_dir=config.cache_dir,
            use_local_data=config.use_local_data,
            data_root=config.data_root,
            subsets=config.subsets,
            files_per_subset=config.files_per_subset,
            use_weighted_files=config.use_weighted_files,
            weighted_file_list=config.weighted_file_list,
            sampling_seed=config.sampling_seed,
            loop_files=config.loop_files,
            debug_log_first_n=config.debug_log_first_n,
            debug_text_words=config.debug_text_words,
        )

        self.batch_size = get_streaming_batch_size(config.batch_size)
        self.shuffle = config.shuffle
        self.shuffle_seed = config.shuffle_seed
        self.shuffle_buffer = config.shuffle_buffer
        if self.shuffle_buffer is None:
            self.shuffle_buffer = 10 * self.batch_size

        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.prefetch_factor = config.prefetch_factor
        self.persistent_workers = config.persistent_workers

        dataset_for_split = cast(Any, self.dataset)
        self.dataset = split_dataset_by_node(  # type: ignore[arg-type]
            dataset_for_split, world_size=num_tasks(), rank=task_id()
        )

        if self.shuffle:
            if hasattr(self.dataset, "shuffle"):
                try:
                    self.dataset = self.dataset.shuffle(
                        buffer_size=self.shuffle_buffer, seed=self.shuffle_seed
                    )
                except TypeError:
                    self.dataset = self.dataset.shuffle(seed=self.shuffle_seed)
            else:
                torch.manual_seed(self.shuffle_seed)
        else:
            torch.manual_seed(self.shuffle_seed)

    def create_dataloader(self):
        data_loader = torch.utils.data.DataLoader(  # type: ignore[attr-defined]
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            prefetch_factor=(
                self.prefetch_factor if self.num_workers > 0 else None
            ),
            persistent_workers=(
                self.persistent_workers if self.num_workers > 0 else False
            ),
            collate_fn=self.data_collator,
        )
        return data_loader
