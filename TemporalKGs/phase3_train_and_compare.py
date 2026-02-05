#!/usr/bin/env python3
"""Phase-3 downstream experiment for ICEWS temporal link prediction.

Compares:
1) Baseline temporal KGE (TTransE-lite)
2) Baseline + influence-aware auxiliary updates from final_influence_graph.json

This script is designed to run on Colab or locally with GPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import DefaultDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TripleSplit:
    h: torch.Tensor
    r: torch.Tensor
    t: torch.Tensor
    time_id: torch.Tensor

    def __len__(self) -> int:
        return int(self.h.shape[0])


class TemporalTransELite(nn.Module):
    def __init__(self, num_entities: int, num_relations: int, num_times: int, emb_dim: int) -> None:
        super().__init__()
        self.entity_emb = nn.Embedding(num_entities, emb_dim)
        self.relation_emb = nn.Embedding(num_relations, emb_dim)
        self.time_emb = nn.Embedding(num_times, emb_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.entity_emb.weight)
        nn.init.xavier_uniform_(self.relation_emb.weight)
        nn.init.xavier_uniform_(self.time_emb.weight)

    def score_triples(
        self,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        t_idx: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> torch.Tensor:
        h = self.entity_emb(h_idx)
        r = self.relation_emb(r_idx)
        tail = self.entity_emb(t_idx)
        time_vec = self.time_emb(time_idx)
        # Higher is better.
        return -(h + r + time_vec - tail).abs().sum(dim=1)

    def score_all_tails(
        self,
        h_idx: torch.Tensor,
        r_idx: torch.Tensor,
        time_idx: torch.Tensor,
    ) -> torch.Tensor:
        query = self.entity_emb(h_idx) + self.relation_emb(r_idx) + self.time_emb(time_idx)
        return -torch.cdist(query, self.entity_emb.weight, p=1.0)


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    default_data_dir = script_dir / "icews05-15_aug_inverse_time_year"
    default_influence = script_dir / "final_influence_graph.json"
    default_output = script_dir / "phase3_results"

    parser = argparse.ArgumentParser(description="Phase-3 temporal prediction experiment (baseline vs influence-aware).")
    parser.add_argument("--data_dir", type=Path, default=default_data_dir)
    parser.add_argument("--influence_graph", type=Path, default=default_influence)
    parser.add_argument("--output_dir", type=Path, default=default_output)
    parser.add_argument(
        "--run_mode",
        choices=["baseline", "influence", "both"],
        default="both",
        help="Run only baseline, only influence-aware, or both for direct comparison.",
    )
    parser.add_argument("--emb_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--influence_lambda", type=float, default=0.05)
    parser.add_argument("--max_influence_neighbors", type=int, default=5)
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=20000,
        help="Cap number of validation/test queries for faster runs. Use <=0 for full split.",
    )
    parser.add_argument(
        "--log_every_steps",
        type=int,
        default=100,
        help="Print train logs every N optimization steps.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_or_add(mapping: dict[str, int], key: str) -> int:
    value = mapping.get(key)
    if value is None:
        value = len(mapping)
        mapping[key] = value
    return value


def encode_split(
    path: Path,
    entity_to_id: dict[str, int],
    relation_to_id: dict[str, int],
    time_to_id: dict[int, int],
) -> TripleSplit:
    h_idx: list[int] = []
    r_idx: list[int] = []
    t_idx: list[int] = []
    time_idx: list[int] = []

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            h_idx.append(get_or_add(entity_to_id, row["head"]))
            r_idx.append(get_or_add(relation_to_id, row["relation"]))
            t_idx.append(get_or_add(entity_to_id, row["tail"]))
            time_raw = int(row["time_index"])
            time_idx.append(get_or_add(time_to_id, time_raw))

    return TripleSplit(
        h=torch.tensor(h_idx, dtype=torch.long),
        r=torch.tensor(r_idx, dtype=torch.long),
        t=torch.tensor(t_idx, dtype=torch.long),
        time_id=torch.tensor(time_idx, dtype=torch.long),
    )


def load_data(data_dir: Path) -> tuple[TripleSplit, TripleSplit, TripleSplit, dict[str, int], dict[str, int], dict[int, int]]:
    train_path = data_dir / "icews_2005-2015_train_normalized.txt"
    valid_path = data_dir / "icews_2005-2015_valid_normalized.txt"
    test_path = data_dir / "icews_2005-2015_test_normalized.txt"
    for p in (train_path, valid_path, test_path):
        if not p.exists():
            raise FileNotFoundError(f"Missing data file: {p}")

    entity_to_id: dict[str, int] = {}
    relation_to_id: dict[str, int] = {}
    time_to_id: dict[int, int] = {}

    train = encode_split(train_path, entity_to_id, relation_to_id, time_to_id)
    valid = encode_split(valid_path, entity_to_id, relation_to_id, time_to_id)
    test = encode_split(test_path, entity_to_id, relation_to_id, time_to_id)

    print(
        "Loaded splits: "
        f"train={len(train):,}, valid={len(valid):,}, test={len(test):,}, "
        f"entities={len(entity_to_id):,}, relations={len(relation_to_id):,}, times={len(time_to_id):,}"
    )
    return train, valid, test, entity_to_id, relation_to_id, time_to_id


def build_true_tails(*splits: TripleSplit) -> DefaultDict[tuple[int, int, int], set[int]]:
    true_tails: DefaultDict[tuple[int, int, int], set[int]] = defaultdict(set)
    for split in splits:
        h_np = split.h.numpy()
        r_np = split.r.numpy()
        t_np = split.t.numpy()
        time_np = split.time_id.numpy()
        for h, r, tail, ti in zip(h_np, r_np, t_np, time_np, strict=True):
            true_tails[(int(h), int(r), int(ti))].add(int(tail))
    return true_tails


def load_influence_adjacency(
    influence_graph_path: Path,
    entity_to_id: dict[str, int],
    max_neighbors: int,
) -> dict[int, list[tuple[int, float]]]:
    if not influence_graph_path.exists():
        raise FileNotFoundError(f"Influence graph not found: {influence_graph_path}")

    with influence_graph_path.open("r", encoding="utf-8") as f:
        raw_graph = json.load(f)

    adjacency: dict[int, list[tuple[int, float]]] = {}
    total_kept_edges = 0
    total_kept_sources = 0

    for src_name, raw_neighbors in raw_graph.items():
        src_id = entity_to_id.get(src_name)
        if src_id is None:
            continue

        neighbors_sorted = sorted(raw_neighbors.items(), key=lambda kv: kv[1], reverse=True)
        kept_neighbors: list[tuple[int, float]] = []
        for dst_name, weight in neighbors_sorted:
            dst_id = entity_to_id.get(dst_name)
            if dst_id is None or dst_id == src_id:
                continue
            w = float(weight)
            if w <= 0.0:
                continue
            kept_neighbors.append((dst_id, w))
            if max_neighbors > 0 and len(kept_neighbors) >= max_neighbors:
                break

        if kept_neighbors:
            adjacency[src_id] = kept_neighbors
            total_kept_sources += 1
            total_kept_edges += len(kept_neighbors)

    print(
        "Loaded influence adjacency: "
        f"sources={total_kept_sources:,}, edges={total_kept_edges:,}, "
        f"max_neighbors={max_neighbors}"
    )
    return adjacency


def compute_influence_loss(
    model: TemporalTransELite,
    batch_entities_cpu: torch.Tensor,
    influence_adjacency: dict[int, list[tuple[int, float]]],
    device: torch.device,
) -> torch.Tensor:
    if not influence_adjacency:
        return torch.zeros((), device=device)

    src_ids: list[int] = []
    dst_ids: list[int] = []
    weights: list[float] = []

    for src_id in batch_entities_cpu.tolist():
        neighbors = influence_adjacency.get(int(src_id))
        if not neighbors:
            continue
        for dst_id, w in neighbors:
            src_ids.append(int(src_id))
            dst_ids.append(int(dst_id))
            weights.append(float(w))

    if not src_ids:
        return torch.zeros((), device=device)

    src_tensor = torch.tensor(src_ids, dtype=torch.long, device=device)
    dst_tensor = torch.tensor(dst_ids, dtype=torch.long, device=device)
    w_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    diff = model.entity_emb(src_tensor) - model.entity_emb(dst_tensor)
    return (w_tensor * diff.pow(2).sum(dim=1)).mean()


def train_one_condition(
    model: TemporalTransELite,
    train_data: TripleSplit,
    device: torch.device,
    num_entities: int,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    grad_clip: float,
    log_every_steps: int,
    influence_adjacency: dict[int, list[tuple[int, float]]] | None = None,
    influence_lambda: float = 0.0,
) -> None:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    num_train = len(train_data)
    step = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        permutation = torch.randperm(num_train)
        epoch_loss = 0.0
        epoch_base_loss = 0.0
        epoch_influence_loss = 0.0
        total_batches = 0

        model.train()
        for start in range(0, num_train, batch_size):
            end = min(start + batch_size, num_train)
            idx = permutation[start:end]

            h_cpu = train_data.h[idx]
            r_cpu = train_data.r[idx]
            t_cpu = train_data.t[idx]
            time_cpu = train_data.time_id[idx]
            neg_t_cpu = torch.randint(0, num_entities, (len(idx),), dtype=torch.long)

            h = h_cpu.to(device)
            r = r_cpu.to(device)
            t = t_cpu.to(device)
            time_idx = time_cpu.to(device)
            neg_t = neg_t_cpu.to(device)

            pos_scores = model.score_triples(h, r, t, time_idx)
            neg_scores = model.score_triples(h, r, neg_t, time_idx)
            base_loss = -F.logsigmoid(pos_scores).mean() - F.logsigmoid(-neg_scores).mean()

            influence_loss = torch.zeros((), device=device)
            if influence_adjacency and influence_lambda > 0.0:
                batch_entities = torch.unique(torch.cat([h_cpu, t_cpu], dim=0))
                influence_loss = compute_influence_loss(model, batch_entities, influence_adjacency, device)

            loss = base_loss + influence_lambda * influence_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            with torch.no_grad():
                model.entity_emb.weight.data = F.normalize(model.entity_emb.weight.data, p=2, dim=1)
                model.relation_emb.weight.data = F.normalize(model.relation_emb.weight.data, p=2, dim=1)
                model.time_emb.weight.data = F.normalize(model.time_emb.weight.data, p=2, dim=1)

            loss_value = float(loss.detach().cpu())
            base_value = float(base_loss.detach().cpu())
            infl_value = float(influence_loss.detach().cpu())

            epoch_loss += loss_value
            epoch_base_loss += base_value
            epoch_influence_loss += infl_value
            total_batches += 1
            step += 1

            if log_every_steps > 0 and step % log_every_steps == 0:
                print(
                    f"step={step:,} epoch={epoch}/{epochs} "
                    f"loss={loss_value:.4f} base={base_value:.4f} infl={infl_value:.4f}"
                )

        epoch_secs = time.time() - epoch_start
        print(
            f"epoch={epoch}/{epochs} "
            f"avg_loss={epoch_loss / max(total_batches, 1):.4f} "
            f"avg_base={epoch_base_loss / max(total_batches, 1):.4f} "
            f"avg_infl={epoch_influence_loss / max(total_batches, 1):.4f} "
            f"time={epoch_secs:.1f}s"
        )


@torch.no_grad()
def evaluate_hits_at_k(
    model: TemporalTransELite,
    split: TripleSplit,
    true_tails: DefaultDict[tuple[int, int, int], set[int]],
    device: torch.device,
    eval_batch_size: int,
    eval_max_samples: int,
) -> dict[str, float]:
    model.eval()
    n_total = len(split)
    if eval_max_samples > 0:
        n_eval = min(n_total, eval_max_samples)
    else:
        n_eval = n_total

    indices = torch.arange(n_eval, dtype=torch.long)
    hits3 = 0
    hits5 = 0
    hits10 = 0
    mrr_sum = 0.0

    for start in range(0, n_eval, eval_batch_size):
        end = min(start + eval_batch_size, n_eval)
        batch_idx = indices[start:end]

        h_cpu = split.h[batch_idx]
        r_cpu = split.r[batch_idx]
        t_cpu = split.t[batch_idx]
        time_cpu = split.time_id[batch_idx]

        h = h_cpu.to(device)
        r = r_cpu.to(device)
        t = t_cpu.to(device)
        time_idx = time_cpu.to(device)

        scores = model.score_all_tails(h, r, time_idx)

        batch_size = int(scores.shape[0])
        for i in range(batch_size):
            key = (int(h_cpu[i]), int(r_cpu[i]), int(time_cpu[i]))
            true_tail = int(t_cpu[i])
            for other_true_tail in true_tails[key]:
                if other_true_tail != true_tail:
                    scores[i, other_true_tail] = -1e9

        true_scores = scores[torch.arange(batch_size, device=device), t]
        ranks = 1 + torch.sum(scores > true_scores.unsqueeze(1), dim=1)
        ranks_cpu = ranks.detach().cpu()

        hits3 += int((ranks_cpu <= 3).sum().item())
        hits5 += int((ranks_cpu <= 5).sum().item())
        hits10 += int((ranks_cpu <= 10).sum().item())
        mrr_sum += float((1.0 / ranks_cpu.float()).sum().item())

    denom = float(n_eval)
    return {
        "num_eval_samples": int(n_eval),
        "hits@3": hits3 / denom,
        "hits@5": hits5 / denom,
        "hits@10": hits10 / denom,
        "mrr": mrr_sum / denom,
    }


def run_condition(
    condition_name: str,
    train_data: TripleSplit,
    valid_data: TripleSplit,
    test_data: TripleSplit,
    true_tails: DefaultDict[tuple[int, int, int], set[int]],
    num_entities: int,
    num_relations: int,
    num_times: int,
    device: torch.device,
    args: argparse.Namespace,
    influence_adjacency: dict[int, list[tuple[int, float]]] | None,
    influence_lambda: float,
) -> dict[str, object]:
    print(f"\n=== Running condition: {condition_name} ===")
    model = TemporalTransELite(
        num_entities=num_entities,
        num_relations=num_relations,
        num_times=num_times,
        emb_dim=args.emb_dim,
    ).to(device)

    train_one_condition(
        model=model,
        train_data=train_data,
        device=device,
        num_entities=num_entities,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        log_every_steps=args.log_every_steps,
        influence_adjacency=influence_adjacency,
        influence_lambda=influence_lambda,
    )

    print(f"Evaluating {condition_name} on valid split...")
    valid_metrics = evaluate_hits_at_k(
        model=model,
        split=valid_data,
        true_tails=true_tails,
        device=device,
        eval_batch_size=args.eval_batch_size,
        eval_max_samples=args.eval_max_samples,
    )
    print(f"valid metrics: {valid_metrics}")

    print(f"Evaluating {condition_name} on test split...")
    test_metrics = evaluate_hits_at_k(
        model=model,
        split=test_data,
        true_tails=true_tails,
        device=device,
        eval_batch_size=args.eval_batch_size,
        eval_max_samples=args.eval_max_samples,
    )
    print(f"test metrics: {test_metrics}")

    return {
        "condition": condition_name,
        "influence_lambda": influence_lambda,
        "valid": valid_metrics,
        "test": test_metrics,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_data, valid_data, test_data, entity_to_id, relation_to_id, time_to_id = load_data(args.data_dir)
    true_tails = build_true_tails(train_data, valid_data, test_data)

    num_entities = len(entity_to_id)
    num_relations = len(relation_to_id)
    num_times = len(time_to_id)

    baseline_result: dict[str, object] | None = None
    influence_result: dict[str, object] | None = None

    if args.run_mode in {"baseline", "both"}:
        baseline_result = run_condition(
            condition_name="baseline",
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            true_tails=true_tails,
            num_entities=num_entities,
            num_relations=num_relations,
            num_times=num_times,
            device=device,
            args=args,
            influence_adjacency=None,
            influence_lambda=0.0,
        )

    if args.run_mode in {"influence", "both"}:
        influence_adjacency = load_influence_adjacency(
            influence_graph_path=args.influence_graph,
            entity_to_id=entity_to_id,
            max_neighbors=args.max_influence_neighbors,
        )
        influence_result = run_condition(
            condition_name="influence_aware",
            train_data=train_data,
            valid_data=valid_data,
            test_data=test_data,
            true_tails=true_tails,
            num_entities=num_entities,
            num_relations=num_relations,
            num_times=num_times,
            device=device,
            args=args,
            influence_adjacency=influence_adjacency,
            influence_lambda=args.influence_lambda,
        )

    result: dict[str, object] = {
        "config": {
            "data_dir": str(args.data_dir),
            "influence_graph": str(args.influence_graph),
            "run_mode": args.run_mode,
            "emb_dim": args.emb_dim,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "eval_max_samples": args.eval_max_samples,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "influence_lambda": args.influence_lambda,
            "max_influence_neighbors": args.max_influence_neighbors,
            "seed": args.seed,
            "device": str(device),
        },
        "dataset_stats": {
            "train": len(train_data),
            "valid": len(valid_data),
            "test": len(test_data),
            "entities": num_entities,
            "relations": num_relations,
            "times": num_times,
        },
        "results": {},
    }

    if baseline_result is not None:
        result["results"]["baseline"] = baseline_result
    if influence_result is not None:
        result["results"]["influence_aware"] = influence_result

    if baseline_result is not None and influence_result is not None:
        delta = {}
        for metric in ("hits@3", "hits@5", "hits@10", "mrr"):
            baseline_value = float(baseline_result["test"][metric])  # type: ignore[index]
            influence_value = float(influence_result["test"][metric])  # type: ignore[index]
            delta[metric] = influence_value - baseline_value
        result["results"]["delta_test"] = delta
        print(f"\nTest delta (influence - baseline): {delta}")

    out_file = args.output_dir / "phase3_results.json"
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved results to: {out_file}")


if __name__ == "__main__":
    main()
