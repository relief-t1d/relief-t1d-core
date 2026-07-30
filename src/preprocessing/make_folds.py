import pandas as pd
import os
import random
from collections import Counter
from typing import Dict, List

# Type alias for clarity: each fold is a dict with three lists of patient IDs (str)
Fold = Dict[str, List[str]]  # {'train': [...], 'val': [...], 'test': [...]}

def make_folds_greedy_splits_valanced_priority(patient_windows: Dict[str, int],
                                               k: int = 5,
                                               seed: int = 42,
                                               ) -> List[Fold]:
    """Balanced *k*-fold split (patient‑wise).

    • **Test assignment (unchanged)** – patients are sorted globally by window
      count ↓, then taken in *k*-sized blocks; one patient from each block is
      randomly assigned to each test fold so that big patients are spread
      evenly.  Remaining (< *k*) patients go to the fold with the lightest
      load.

    • **Train / Val assignment (NEW)** – the remaining patients of each fold
      are also processed in blocks of *k* **(8 → 7‑train / 1‑val)**.  The last
      block, even if smaller than *k*, still sends **one random patient to
      validation** and the rest to training.  This guarantees *at least one*
      validation patient per fold.
    """

    rng = random.Random(seed)

    # ------------------------------------------------------------------
    # 1️⃣  Sort all patients by window count (descending)
    # ------------------------------------------------------------------
    sorted_patients = sorted(patient_windows.keys(),
                             key=lambda pid: patient_windows[pid],
                             reverse=True)

    # ------------------------------------------------------------------
    # 2️⃣  Assign test patients block‑wise (size == k)
    # ------------------------------------------------------------------
    test_folds: List[List[str]] = [[] for _ in range(k)]
    fold_loads: List[int] = [0] * k

    block_size = k  # 5 if k == 5 (one per fold)
    idx = 0
    while idx + block_size <= len(sorted_patients):
        block = sorted_patients[idx: idx + block_size]
        block.sort(key=lambda pid: patient_windows[pid], reverse=True)
        rng.shuffle(block)  # random perm in the block
        for fold_idx, patient_id in enumerate(block):
            test_folds[fold_idx].append(patient_id)
            fold_loads[fold_idx] += patient_windows[patient_id]
        idx += block_size

    # Remaining (< k) patients – assign to lightest fold
    for pid in sorted_patients[idx:]:
        lightest = fold_loads.index(min(fold_loads))
        test_folds[lightest].append(pid)
        fold_loads[lightest] += patient_windows[pid]

    # ------------------------------------------------------------------
    # 3️⃣  Build train / val with 7‑train / 1‑val blocks (size == 8)
    # ------------------------------------------------------------------
    all_ids_set = set(patient_windows.keys())
    folds: List[Fold] = []
    train_block = 7  # patients for train per block of 8
    val_block = 1  # patients for val  per block of 8
    block_tv = train_block + val_block  # 8

    for fold_idx in range(k):
        test_ids = set(test_folds[fold_idx])
        remaining_ids = [pid for pid in sorted_patients if pid not in test_ids]

        # --- process in blocks of 8 ------------------------------------
        train_ids, val_ids = [], []
        ptr = 0
        n_remaining = len(remaining_ids)

        while ptr + block_tv <= n_remaining:
            block = remaining_ids[ptr: ptr + block_tv]
            # already sorted globally; still shuffle to randomise who goes to val
            rng.shuffle(block)
            val_ids.append(block[0])  # 1 patient → val
            train_ids.extend(block[1:])  # 7 patients → train
            ptr += block_tv

        # Handle last (incomplete) block → send 1 to val, rest to train
        tail = remaining_ids[ptr:]
        if tail:
            rng.shuffle(tail)
            val_ids.append(tail[0])
            train_ids.extend(tail[1:])

        folds.append({
            'train': sorted(train_ids),
            'val': sorted(val_ids),
            'test': sorted(test_folds[fold_idx])
        })

    return folds

def validate_folds(folds: List[Fold], patient_measurements: Dict[str, int], tolerance_pct: float = 10.0,
                   horizon: int = 2) -> None:
    """
    Validate a list of patient-wise cross-validation folds.
    """
    print(f"\nValidating folds for horizon: {horizon}...")

    all_patients = set(patient_measurements)

    # ------------------------------------------------------------------ R1/R3
    for idx, fold in enumerate(folds):
        tr, va, te = map(set, (fold['train'], fold['val'], fold['test']))

        # R1 – exclusivity within the same fold
        assert tr.isdisjoint(va) and tr.isdisjoint(te) and va.isdisjoint(te), (
            f"[R1] Overlap detected in fold {idx}"
        )

        # R3 – union covers the entire cohort
        assert len(tr | va | te) == len(all_patients), (
            f"[R3] Missing patients in fold {idx}"
        )

    print("\n ✓ R1 and R3 passed for every fold.")

    # ------------------------------------------------------------------ R2
    test_counter = Counter(pid for fold in folds for pid in fold['test'])
    duplicates = [pid for pid, cnt in test_counter.items() if cnt != 1]

    assert not duplicates, (
        f"[R2] Patients appearing in test 0 or >1 times: {duplicates}"
    )
    print("✓ R2 passed – each patient appears in *test* exactly once.")

    # ------------------------------------------------------------------ R4 (informative)
    def test_load(ids: List[str]) -> int:
        """Total number of measurements for this group of patients."""
        return sum(patient_measurements[pid] for pid in ids)

    loads = [test_load(f['test']) for f in folds]
    mean_load = sum(loads) / len(loads)
    pct_dev = [100 * (l - mean_load) / mean_load for l in loads]

    print("\nTest-set load per fold (number of measurements):", loads)
    print("Deviation from mean (%)                      :", [round(p, 2) for p in pct_dev])

    if any(abs(p) > tolerance_pct for p in pct_dev):
        print(f"⚠️  [R4] Warning: at least one fold deviates by more than {tolerance_pct}% "
              "from the mean test load.")
    else:
        print(f"✓ R4 passed – test loads within ±{tolerance_pct}% of the mean.")

    print(f"\nAll mandatory validation checks for horizon {horizon} completed successfully.")


def tag_folds_and_save(
        df_windows: pd.DataFrame,
        folds_splits: List[Fold],
        output_path: str,
        dataset_name: str,
        historic: str,
        horizon: int,
        k_folds: int = 5,
        column_prefix: str = "fold") -> None:
    """
    Add one column per fold (fold0 … fold{k-1}) indicating whether each window
    belongs to 'train', 'val' or 'test' in that fold, then write a single
    Parquet file.
    """
    df_tagged = df_windows.copy()

    for idx, fold in enumerate(folds_splits):
        # Build a mapping patient_id -> split label for this fold
        label_map = {pid: "train" for pid in fold["train"]}
        label_map.update({pid: "val" for pid in fold["val"]})
        label_map.update({pid: "test" for pid in fold["test"]})

        col_name = f"{column_prefix}{idx}"
        df_tagged[col_name] = df_tagged["patient_id"].map(label_map).astype("category")

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    df_tagged.to_parquet(f'{output_path}{dataset_name}_H{historic}_PH{horizon}_kfold{k_folds}.parquet', index=False)
    print(f"\n ✓ File {dataset_name}_horizon_{horizon} written: {output_path}"
          f"({len(df_windows):,} rows, {len(folds_splits)} fold-columns)")

    return df_tagged