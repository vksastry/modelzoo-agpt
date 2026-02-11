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

"""HuggingFace OLMo Mix dataset loader."""

import bisect
import gzip
import io
import json
import os
import random
from glob import glob
from itertools import zip_longest
from typing import Any, Iterable, Iterator, Optional, Sequence, cast

from datasets import Dataset, IterableDataset, load_dataset
from huggingface_hub import hf_hub_download, hf_hub_url, list_repo_files
from transformers import AutoTokenizer

from cerebras.modelzoo.data_preparation.huggingface.CSDataCollatorForLanguageModeling import (
    CSDataCollatorForLanguageModeling,
)

# Suppress warnings about using fast tokenizers
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"


def _load_olmomix_from_data_files(
    subset: str,
    split: str,
    streaming: bool,
    max_files: Optional[int],
    data_files: Optional[Any],
    cache_dir: Optional[str],
):
    repo_id = "allenai/olmo-mix-1124"

    if data_files is None:
        prefix = f"data/{subset}/"
        files = list_repo_files(repo_id, repo_type="dataset")
        candidates = [f for f in files if f.startswith(prefix)]
        candidates = [
            f
            for f in candidates
            if f.endswith(
                (
                    ".json",
                    ".jsonl",
                    ".jsonl.gz",
                    ".json.gz",
                    ".parquet",
                )
            )
        ]
        candidates.sort()
        if max_files is not None:
            candidates = candidates[:max_files]
        if not candidates:
            raise ValueError(
                f"No data files found for subset '{subset}' under {prefix}."
            )
        if streaming:
            data_files = [
                hf_hub_url(repo_id, filename=f, repo_type="dataset")
                for f in candidates
            ]
        else:
            data_files = [
                hf_hub_download(
                    repo_id,
                    filename=f,
                    repo_type="dataset",
                    cache_dir=cache_dir,
                )
                for f in candidates
            ]

    first_file = data_files[0] if isinstance(data_files, list) else None
    if first_file and str(first_file).endswith(".parquet"):
        builder = "parquet"
    else:
        builder = "json"

    return load_dataset(
        builder,
        data_files={split: data_files} if isinstance(data_files, list) else data_files,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )


def _iter_jsonl_from_path(path: str) -> Iterable[dict]:
    if path.endswith(".json.zstd") or path.endswith(".jsonl.zstd"):
        import zstandard as zstd

        with open(path, "rb") as f:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(f) as reader:
                text = io.TextIOWrapper(reader, encoding="utf-8")
                for line in text:
                    line = line.strip()
                    if not line:
                        continue
                    yield json.loads(line)
    elif path.endswith(".json.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def _collect_subset_files(
    data_root: str,
    subset: str,
    files_per_subset: Optional[int],
) -> Sequence[str]:
    subset_dir = os.path.join(data_root, subset)
    patterns = [
        os.path.join(subset_dir, "*.json.zstd"),
        os.path.join(subset_dir, "*.json.gz"),
        os.path.join(subset_dir, "*.jsonl"),
        os.path.join(subset_dir, "*.json"),
    ]
    files = []
    for pattern in patterns:
        files.extend(sorted(glob(pattern)))
    if files_per_subset is not None:
        files = files[:files_per_subset]
    return files


def _resolve_subsets(data_root: str, subsets: Sequence[str] | str) -> Sequence[str]:
    if isinstance(subsets, str) and subsets.lower() == "all":
        candidates = [
            d
            for d in os.listdir(data_root)
            if os.path.isdir(os.path.join(data_root, d))
        ]
        valid = []
        for subset in sorted(candidates):
            files = _collect_subset_files(data_root, subset, files_per_subset=1)
            if files:
                valid.append(subset)
        return valid
    if isinstance(subsets, str):
        return [subsets]
    return list(subsets)


def _load_weighted_file_list(weighted_file_list: str) -> list[tuple[float, str]]:
    entries = []
    with open(weighted_file_list, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                raise ValueError(
                    "Weighted file list must have at least weight and path"
                )
            weight = float(parts[0])
            path = parts[1]
            if not os.path.exists(path):
                for suffix in (
                    ".json.zstd",
                    ".jsonl.zstd",
                    ".json.gz",
                    ".jsonl",
                    ".json",
                ):
                    candidate = f"{path}{suffix}"
                    if os.path.exists(candidate):
                        path = candidate
                        break
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Weighted file list path not found: {path}"
                )
            if weight <= 0:
                continue
            entries.append((weight, path))

    if not entries:
        raise ValueError("Weighted file list is empty after parsing.")
    total = sum(w for w, _ in entries)
    if total <= 0:
        raise ValueError("Sum of weights must be positive.")
    normalized = [(w / total, p) for w, p in entries]
    return normalized


def _weighted_file_generator(
    weighted_entries: Sequence[tuple[float, str]],
    sampling_seed: int,
    loop_files: bool,
    debug_log_first_n: int,
    debug_text_words: int,
) -> Iterable[dict]:
    weights = [w for w, _ in weighted_entries]
    paths = [p for _, p in weighted_entries]
    cdf = []
    total = 0.0
    for w in weights:
        total += w
        cdf.append(total)

    rng = random.Random(sampling_seed)
    iterators: dict[str, Iterator[dict]] = {}

    seen = 0
    while True:
        r = rng.random() * total
        idx = bisect.bisect_left(cdf, r)
        if idx >= len(paths):
            idx = len(paths) - 1
        path = paths[idx]

        it = iterators.get(path)
        if it is None:
            it = iter(_iter_jsonl_from_path(path))
            iterators[path] = it

        try:
            obj = next(cast(Iterator[dict], it))
        except StopIteration:
            if not loop_files:
                raise
            it = iter(_iter_jsonl_from_path(path))
            iterators[path] = it
            obj = next(cast(Iterator[dict], it))

        if debug_log_first_n > 0 and seen < debug_log_first_n:
            preview = ""
            text = obj.get("text") if isinstance(obj, dict) else None
            if isinstance(text, str):
                words = text.split()
                preview = " ".join(words[:debug_text_words])
            print(
                f"[weighted_sampling] sample {seen + 1} file={path} text=\"{preview}\""
            )
        seen += 1
        yield obj


def _build_local_iterable_dataset(
    data_root: str,
    subsets: Sequence[str] | str,
    files_per_subset: Optional[int],
    cache_dir: Optional[str],
) -> IterableDataset:
    subset_list = _resolve_subsets(data_root, subsets)
    per_subset_files = [
        _collect_subset_files(data_root, subset, files_per_subset)
        for subset in subset_list
    ]
    interleaved_files = []
    for group in zip_longest(*per_subset_files):
        for path in group:
            if path is not None:
                interleaved_files.append(path)

    def generator():
        for path in interleaved_files:
            for obj in _iter_jsonl_from_path(path):
                if "text" in obj:
                    yield {"text": obj["text"]}

    if cache_dir is None:
        cache_dir = "/tmp/hf_cache"

    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    return IterableDataset.from_generator(generator)


def _build_weighted_iterable_dataset(
    weighted_file_list: str,
    sampling_seed: int,
    loop_files: bool,
    cache_dir: Optional[str],
    debug_log_first_n: int,
    debug_text_words: int,
) -> IterableDataset:
    if cache_dir is None:
        cache_dir = "/tmp/hf_cache"
    os.makedirs(cache_dir, exist_ok=True)
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["HF_HOME"] = cache_dir
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)

    weighted_entries = _load_weighted_file_list(weighted_file_list)

    def generator():
        for obj in _weighted_file_generator(
            weighted_entries,
            sampling_seed,
            loop_files,
            debug_log_first_n,
            debug_text_words,
        ):
            if "text" in obj:
                yield {"text": obj["text"]}

    return IterableDataset.from_generator(generator)


def HuggingFace_OlmoMix(
    split: str = "train",
    subset: str = "default",
    num_workers: int = 0,
    sequence_length: int = 2048,
    tokenizer_name: str = "gpt2",
    max_samples: Optional[int] = 10000,
    streaming: bool = True,
    use_data_files: bool = True,
    max_files: Optional[int] = 2,
    data_files: Optional[Any] = None,
    cache_dir: Optional[str] = None,
    use_local_data: bool = False,
    data_root: Optional[str] = None,
    subsets: Sequence[str] | str = "wiki",
    files_per_subset: Optional[int] = 2,
    use_weighted_files: bool = False,
    weighted_file_list: Optional[str] = None,
    sampling_seed: int = 0,
    loop_files: bool = True,
    debug_log_first_n: int = 0,
    debug_text_words: int = 10,
):
    if use_weighted_files:
        if weighted_file_list is None:
            raise ValueError(
                "weighted_file_list must be set when use_weighted_files is True"
            )
        dataset = _build_weighted_iterable_dataset(
            weighted_file_list=weighted_file_list,
            sampling_seed=sampling_seed,
            loop_files=loop_files,
            cache_dir=cache_dir,
            debug_log_first_n=debug_log_first_n,
            debug_text_words=debug_text_words,
        )
    elif use_local_data:
        if data_root is None:
            raise ValueError("data_root must be set when use_local_data is True")
        if cache_dir is None:
            cache_dir = "/tmp/hf_cache"
        dataset = _build_local_iterable_dataset(
            data_root=data_root,
            subsets=subsets,
            files_per_subset=files_per_subset,
            cache_dir=cache_dir,
        )
    elif use_data_files:
        dataset = _load_olmomix_from_data_files(
            subset=subset,
            split=split,
            streaming=streaming,
            max_files=max_files,
            data_files=data_files,
            cache_dir=cache_dir,
        )
    else:
        dataset = load_dataset(
            "allenai/olmo-mix-1124",
            name=subset,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
        )

    dataset = cast(Dataset | IterableDataset, dataset)

    if max_samples is not None:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(max_samples)
            if streaming:
                dataset = Dataset.from_list(list(dataset))
        elif isinstance(dataset, Dataset):
            dataset = dataset.select(range(max_samples))

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
    tokenizer.add_bos_token = (
        False  # BOS token added in CSDataCollatorForLanguageModeling
    )

    def preprocess_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=sequence_length,
        )

    remove_columns = getattr(dataset, "column_names", None)
    if not remove_columns:
        remove_columns = ["text"]
    if not streaming and isinstance(dataset, Dataset):
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            num_proc=num_workers,
            remove_columns=remove_columns,
        )
    else:
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=remove_columns,
        )

    def group_texts(examples):
        token_keys = []
        for k, v in examples.items():
            if not v:
                continue
            first = v[0]
            if isinstance(first, list):
                token_keys.append(k)
            elif isinstance(first, int):
                token_keys.append(k)

        if not token_keys:
            return {}

        concatenated_examples = {k: sum(examples[k], []) for k in token_keys}
        total_length = len(concatenated_examples[token_keys[0]])
        if total_length >= sequence_length:
            total_length = (total_length // sequence_length) * sequence_length
        result = {
            k: [
                t[i : i + sequence_length]
                for i in range(0, total_length, sequence_length)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    dataset = tokenized_dataset.map(group_texts, batched=True)

    tokenizer.pad_token = tokenizer.eos_token

    data_collator = CSDataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )
    return dataset, data_collator
