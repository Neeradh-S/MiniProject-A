#!/usr/bin/env python3
"""
GNN-based Genome Assembly Pipeline
====================================
Full pipeline covering:
  1. De Bruijn graph construction from reads + reference labeling
  2. Feature engineering (extended)
  3. GNN model definition (GraphSAGE + edge MLP)
  4. Training phase  (requires reference.fasta)
  5. Inference phase (no reference needed)
  6. Assembly        (Eulerian path -> contigs)
  7. Evaluation      (N50, contig stats)

Usage:
  # Train
  python3 training2.py --mode train \
      --reference reference.fasta \
      --reads     reads.fastq \
      --model     model.pt

  # Inference (no reference)
  python3 training2.py --mode infer \
      --reads  new_reads.fastq \
      --model  model.pt \
      --output contigs.fasta
"""

import os
import re
import math
import argparse
import numpy as np
import networkx as nx
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.nn as dglnn

os.makedirs("output", exist_ok=True)


# ============================================================
# SECTION 1 — PARSERS
# ============================================================

def parse_fasta(path):
    sequences, parts = [], []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if parts:
                    sequences.append("".join(parts).upper())
                    parts = []
            else:
                parts.append(line)
        if parts:
            sequences.append("".join(parts).upper())
    return sequences


def parse_fastq(path):
    sequences, parts = [], []
    state = "HEADER"
    with open(path) as fh:
        for raw in fh:
            line = raw.rstrip()
            if not line:
                continue
            if state == "HEADER":
                if line.startswith("@"):
                    parts = []
                    state = "SEQUENCE"
            elif state == "SEQUENCE":
                if line.startswith("+"):
                    if parts:
                        sequences.append("".join(parts).upper())
                    state = "QUALITY"
                else:
                    parts.append(line)
            elif state == "QUALITY":
                if line.startswith("@"):
                    parts = []
                    state = "SEQUENCE"
    return sequences


# ============================================================
# SECTION 2 — K-MER COUNTING + FILTER
# ============================================================

def count_kmers(sequences, k):
    counts = Counter()
    valid = re.compile(r'^[ACGT]+$')
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            kmer = seq[i: i + k]
            if valid.match(kmer):
                counts[kmer] += 1
    return counts


def filter_kmers(kmer_counts, min_count=3):
    """
    Remove low-abundance k-mers that are almost certainly sequencing errors.

    With depth=60:
      Real genomic k-mers  appear ~60 times  -> kept
      Sequencing error k-mers appear 1-2 times -> removed

    This reduces nodes from ~26000 down to ~1400 for a 1441bp gene,
    dramatically improving training precision and F1 score.
    """
    before = len(kmer_counts)
    filtered = Counter({kmer: cnt for kmer, cnt in kmer_counts.items()
                        if cnt >= min_count})
    after = len(filtered)
    removed = before - after
    print("    k-mer filter     : " + str(before) + " -> " + str(after) +
          "  (removed " + str(removed) + " low-abundance error k-mers)")
    return filtered


# ============================================================
# SECTION 3 — FEATURE HELPERS
# ============================================================

def gc_content(seq):
    if not seq:
        return 0.0
    return (seq.count("G") + seq.count("C")) / len(seq)


def sequence_entropy(seq):
    if not seq:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


# ============================================================
# SECTION 4 — DE BRUIJN GRAPH BUILDER
# ============================================================

class DeBruijnGraphBuilder:
    BASES = ("A", "T", "C", "G")

    def __init__(self, kmer_counts, k, reference_sequences=None):
        self.k = k
        self.kmer_counts = kmer_counts
        self.reference_sequences = reference_sequences or []

        self.kmer_to_idx = {
            kmer: idx for idx, kmer in enumerate(kmer_counts.keys())
        }
        self.nodes = {}
        self.edges = {}

    def _label_edge(self, edge_seq):
        for ref in self.reference_sequences:
            if edge_seq in ref:
                return 1
        return 0

    def _build_nodes(self):
        print("  Nodes: " + str(len(self.kmer_to_idx)))
        for kmer, idx in self.kmer_to_idx.items():
            self.nodes[idx] = {"kmer": kmer, "abundance": self.kmer_counts[kmer]}

    def _node_degree(self):
        in_deg  = Counter()
        out_deg = Counter()
        for src, dst in self.edges:
            out_deg[src] += 1
            in_deg[dst]  += 1
        return {
            idx: {"in": in_deg[idx], "out": out_deg[idx]}
            for idx in self.nodes
        }

    def _build_edges(self):
        print("  Building edges ...")
        kmer_set = self.kmer_to_idx
        for kmer, src_idx in kmer_set.items():
            suffix = kmer[1:]
            for base in self.BASES:
                neighbor = suffix + base
                if neighbor in kmer_set:
                    dst_idx  = kmer_set[neighbor]
                    edge_seq = kmer + base
                    src_ab   = self.kmer_counts[kmer]
                    dst_ab   = self.kmer_counts[neighbor]
                    self.edges[(src_idx, dst_idx)] = {
                        "edge_seq":       edge_seq,
                        "abundance_avg":  float((src_ab + dst_ab) / 2.0),
                        "occurrence_sim": float(abs(src_ab - dst_ab)),
                        "gc":             float(gc_content(edge_seq)),
                        "entropy":        float(sequence_entropy(edge_seq)),
                        "label":          float(self._label_edge(edge_seq)),
                    }
        print("  Edges: " + str(len(self.edges)))

    def _build_dgl_graph(self):
        print("  Assembling DGL graph ...")
        DiG = nx.DiGraph()

        degrees = self._node_degree()
        for idx, attrs in self.nodes.items():
            deg = degrees.get(idx, {"in": 0, "out": 0})
            DiG.add_node(
                idx,
                x=float(attrs["abundance"]),
                in_deg=float(deg["in"]),
                out_deg=float(deg["out"]),
            )

        for (src, dst), ea in self.edges.items():
            DiG.add_edge(
                src, dst,
                e=[ea["abundance_avg"], ea["occurrence_sim"],
                   ea["gc"], ea["entropy"]],
                y=ea["label"],
            )

        g = dgl.from_networkx(
            DiG,
            node_attrs=["x", "in_deg", "out_deg"],
            edge_attrs=["e", "y"],
        )

        # Z-score normalise node features
        node_mat = torch.stack(
            [g.ndata["x"], g.ndata["in_deg"], g.ndata["out_deg"]], dim=1
        ).float()
        n_mean = node_mat.mean(dim=0)
        n_std  = node_mat.std(dim=0).clamp(min=1e-9)
        g.ndata["x"] = torch.round((node_mat - n_mean) / n_std * 10000) / 10000
        del g.ndata["in_deg"], g.ndata["out_deg"]

        # Z-score normalise edge features
        ef = g.edata["e"].float()
        ef_mean = ef.mean(dim=0)
        ef_std  = ef.std(dim=0).clamp(min=1e-9)
        g.edata["e"] = torch.round((ef - ef_mean) / ef_std * 10000) / 10000

        return g

    def build(self):
        self._build_nodes()
        self._build_edges()
        return self._build_dgl_graph()


# ============================================================
# SECTION 5 — GNN MODEL
# ============================================================

class GenomeAssemblyGNN(nn.Module):

    def __init__(self, in_node_feats=3, in_edge_feats=4,
                 hidden_dim=64, n_layers=2, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.convs = nn.ModuleList()
        dims = [in_node_feats] + [hidden_dim] * n_layers
        for i in range(n_layers):
            self.convs.append(
                dglnn.SAGEConv(dims[i], dims[i + 1], aggregator_type="mean")
            )

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + in_edge_feats, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_nodes(self, g, x):
        h = x
        for conv in self.convs:
            h = conv(g, h)
            h = F.relu(h)
            h = self.dropout(h)
        return h

    def forward(self, g, node_feats, edge_feats):
        h = self.encode_nodes(g, node_feats)
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(
                lambda edges: {
                    "cat": torch.cat(
                        [edges.src["h"], edges.dst["h"], edge_feats], dim=1
                    )
                }
            )
            logits = self.edge_mlp(g.edata["cat"])
        return logits


# ============================================================
# SECTION 6 — TRAINING
# ============================================================

def train(
    reference_fasta,
    reads_fastq,
    model_path,
    k=21,
    hidden_dim=64,
    n_layers=2,
    dropout=0.3,
    lr=1e-3,
    epochs=100,
    threshold=0.5,
    min_kmer_count=3,
):
    print("=" * 60)
    print("PHASE 1 — TRAINING")
    print("=" * 60)

    print("\n[1] Parsing files ...")
    ref_seqs  = parse_fasta(reference_fasta)
    read_seqs = parse_fastq(reads_fastq)
    print("    Reference sequences : " + str(len(ref_seqs)))
    print("    Reads               : " + str(len(read_seqs)))

    print("\n[2] Counting " + str(k) + "-mers ...")
    kmer_counts = count_kmers(read_seqs, k)
    kmer_counts = filter_kmers(kmer_counts, min_count=min_kmer_count)
    print("    Unique k-mers       : " + str(len(kmer_counts)))

    print("\n[3] Building De Bruijn graph ...")
    builder = DeBruijnGraphBuilder(kmer_counts, k, ref_seqs)
    g = builder.build()

    node_feats = g.ndata["x"].float()
    edge_feats = g.edata["e"].float()
    labels     = g.edata["y"].float()

    n_pos = labels.sum().item()
    n_neg = len(labels) - n_pos
    print("\n[4] Label distribution:")
    print("    Positive (y=1) : " + str(int(n_pos)) +
          " (" + str(round(100 * n_pos / max(len(labels), 1), 1)) + "%)")
    print("    Negative (y=0) : " + str(int(n_neg)) +
          " (" + str(round(100 * n_neg / max(len(labels), 1), 1)) + "%)")

    pos_weight = torch.tensor([n_neg / max(n_pos, 1)])
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    in_node = node_feats.shape[1]
    in_edge = edge_feats.shape[1]
    model   = GenomeAssemblyGNN(in_node, in_edge, hidden_dim, n_layers, dropout)
    optim   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, patience=10, factor=0.5
    )

    print("\n[5] Training for " + str(epochs) + " epochs ...\n")
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        logits = model(g, node_feats, edge_feats).squeeze()
        loss   = loss_fn(logits, labels)

        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step(loss)

        if epoch % 10 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > threshold).float()
                acc   = (preds == labels).float().mean().item()
                tp    = ((preds == 1) & (labels == 1)).sum().item()
                fp    = ((preds == 1) & (labels == 0)).sum().item()
                fn    = ((preds == 0) & (labels == 1)).sum().item()
                prec  = tp / max(tp + fp, 1)
                rec   = tp / max(tp + fn, 1)
                f1    = 2 * prec * rec / max(prec + rec, 1e-9)

            print(
                "  Epoch " + str(epoch).rjust(4) +
                " | Loss " + str(round(loss.item(), 4)) +
                " | Acc "  + str(round(acc, 4)) +
                " | Prec " + str(round(prec, 4)) +
                " | Rec "  + str(round(rec, 4)) +
                " | F1 "   + str(round(f1, 4))
            )

            if loss.item() < best_loss:
                best_loss = loss.item()
                torch.save(
                    {
                        "model_state":   model.state_dict(),
                        "in_node_feats": in_node,
                        "in_edge_feats": in_edge,
                        "hidden_dim":    hidden_dim,
                        "n_layers":      n_layers,
                        "dropout":       dropout,
                        "k":             k,
                        "threshold":     threshold,
                        "min_kmer_count": min_kmer_count,
                    },
                    model_path,
                )
                print("           -> Model saved  (loss=" +
                      str(round(best_loss, 4)) + ")")

    dgl.save_graphs("output/train_graph.bin", [g])
    _save_label_stats(builder.edges, "output/train_label_stats.txt")
    print("\nTraining complete. Best loss: " + str(round(best_loss, 4)))
    print("Model saved to: " + model_path)


# ============================================================
# SECTION 7 — INFERENCE
# ============================================================

def infer(
    reads_fastq,
    model_path,
    output_fasta="output/contigs.fasta",
    threshold=None,
):
    print("=" * 60)
    print("PHASE 2 — INFERENCE  (no reference)")
    print("=" * 60)

    print("\n[1] Loading model ...")
    ckpt = torch.load(model_path, map_location="cpu")
    k               = ckpt["k"]
    thr             = threshold if threshold is not None else ckpt["threshold"]
    min_kmer_count  = ckpt.get("min_kmer_count", 3)

    model = GenomeAssemblyGNN(
        in_node_feats = ckpt["in_node_feats"],
        in_edge_feats = ckpt["in_edge_feats"],
        hidden_dim    = ckpt["hidden_dim"],
        n_layers      = ckpt["n_layers"],
        dropout       = ckpt["dropout"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print("    k=" + str(k) + ", threshold=" + str(thr) +
          ", min_kmer_count=" + str(min_kmer_count))

    print("\n[2] Parsing reads ...")
    read_seqs = parse_fastq(reads_fastq)
    print("    Reads: " + str(len(read_seqs)))

    print("\n[3] Counting " + str(k) + "-mers ...")
    kmer_counts = count_kmers(read_seqs, k)
    kmer_counts = filter_kmers(kmer_counts, min_count=min_kmer_count)
    print("    Unique k-mers: " + str(len(kmer_counts)))

    print("\n[4] Building De Bruijn graph (no reference) ...")
    builder = DeBruijnGraphBuilder(kmer_counts, k, reference_sequences=None)
    g = builder.build()

    node_feats = g.ndata["x"].float()
    edge_feats = g.edata["e"].float()

    print("\n[5] Predicting edge confidence ...")
    with torch.no_grad():
        logits = model(g, node_feats, edge_feats).squeeze()
        probs  = torch.sigmoid(logits)

    n_confident = (probs > thr).sum().item()
    print("    Confident edges (>" + str(thr) + ") : " +
          str(n_confident) + " / " + str(len(probs)))

    print("\n[6] Assembling contigs ...")
    contigs = assemble_contigs(probs, builder, thr)
    print("    Contigs assembled : " + str(len(contigs)))

    _save_contigs(contigs, output_fasta)
    stats = evaluate_assembly(contigs)
    _print_stats(stats)

    return contigs, stats


# ============================================================
# SECTION 8 — ASSEMBLY
# ============================================================

def assemble_contigs(probs, builder, threshold=0.5):
    edge_keys = list(builder.edges.keys())
    confident = probs.numpy() > threshold

    assembly_graph = nx.DiGraph()
    for i, (src, dst) in enumerate(edge_keys):
        if confident[i]:
            edge_seq  = builder.edges[(src, dst)]["edge_seq"]
            edge_prob = float(probs[i])
            assembly_graph.add_edge(src, dst, seq=edge_seq, prob=edge_prob)

    if assembly_graph.number_of_edges() == 0:
        print("  WARNING: No confident edges found. Lower the threshold.")
        return []

    contigs = []
    components = list(nx.weakly_connected_components(assembly_graph))
    print("  Connected components  : " + str(len(components)))

    for comp in components:
        sub = assembly_graph.subgraph(comp).copy()
        if sub.number_of_edges() < 2:
            continue
        try:
            _balance_graph(sub)
            path   = list(nx.eulerian_path(sub))
            contig = _path_to_sequence(path, sub, builder)
            if contig:
                contigs.append(contig)
        except nx.NetworkXError:
            contig = _greedy_traversal(sub, builder)
            if contig:
                contigs.append(contig)

    contigs.sort(key=len, reverse=True)
    return contigs


def _balance_graph(g):
    surplus_out = []
    surplus_in  = []
    for node in g.nodes():
        diff = g.out_degree(node) - g.in_degree(node)
        if diff > 0:
            surplus_out.extend([node] * diff)
        elif diff < 0:
            surplus_in.extend([node] * abs(diff))
    for out_node, in_node in zip(surplus_out, surplus_in):
        g.add_edge(out_node, in_node, seq="", prob=0.0, virtual=True)


def _path_to_sequence(path, graph, builder):
    if not path:
        return ""
    first_node = path[0][0]
    sequence   = builder.nodes[first_node]["kmer"]
    for src, dst in path:
        edge_data = graph[src][dst]
        if edge_data.get("virtual", False):
            continue
        edge_seq = edge_data.get("seq", "")
        if edge_seq:
            sequence += edge_seq[-1]
    return sequence


def _greedy_traversal(graph, builder, min_len=50):
    if graph.number_of_nodes() == 0:
        return ""
    start = max(
        graph.nodes(),
        key=lambda n: graph.out_degree(n) - graph.in_degree(n)
    )
    visited_edges = set()
    sequence      = builder.nodes[start]["kmer"]
    current       = start
    while True:
        candidates = [
            (current, dst, graph[current][dst]["prob"])
            for dst in graph.successors(current)
            if (current, dst) not in visited_edges
        ]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[2], reverse=True)
        _, nxt, _ = candidates[0]
        edge_seq  = graph[current][nxt].get("seq", "")
        if edge_seq:
            sequence += edge_seq[-1]
        visited_edges.add((current, nxt))
        current = nxt
    return sequence if len(sequence) >= min_len else ""


# ============================================================
# SECTION 9 — EVALUATION
# ============================================================

def evaluate_assembly(contigs):
    if not contigs:
        return {}
    lengths = sorted([len(c) for c in contigs], reverse=True)
    total   = sum(lengths)

    cumsum, n50 = 0, 0
    for l in lengths:
        cumsum += l
        if cumsum >= total / 2:
            n50 = l
            break

    cumsum, n90 = 0, 0
    for l in lengths:
        cumsum += l
        if cumsum >= total * 0.9:
            n90 = l
            break

    return {
        "n_contigs":     len(contigs),
        "total_length":  total,
        "longest":       lengths[0],
        "shortest":      lengths[-1],
        "mean_length":   int(np.mean(lengths)),
        "median_length": int(np.median(lengths)),
        "n50":           n50,
        "n90":           n90,
    }


def _print_stats(stats):
    print("\n" + "=" * 60)
    print("ASSEMBLY STATISTICS")
    print("=" * 60)
    for key, val in stats.items():
        print("  " + key.ljust(20) + ": " + str(val))
    print("=" * 60)


# ============================================================
# SECTION 10 — SAVE UTILITIES
# ============================================================

def _save_contigs(contigs, path):
    os.makedirs(
        os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True
    )
    with open(path, "w") as f:
        for i, seq in enumerate(contigs, 1):
            f.write(">contig_" + str(i).zfill(4) +
                    "  len=" + str(len(seq)) + "\n")
            for j in range(0, len(seq), 80):
                f.write(seq[j: j + 80] + "\n")
    print("\n  Contigs saved to: " + path)


def _save_label_stats(edges, path):
    labels = [int(e["label"]) for e in edges.values()]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    with open(path, "w") as f:
        f.write("Total edges   : " + str(len(labels)) + "\n")
        f.write("Positive (y=1): " + str(n_pos) +
                "  (" + str(round(100 * n_pos / max(len(labels), 1), 1)) + "%)\n")
        f.write("Negative (y=0): " + str(n_neg) +
                "  (" + str(round(100 * n_neg / max(len(labels), 1), 1)) + "%)\n")


# ============================================================
# SECTION 11 — CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="GNN-based genome assembler from De Bruijn graphs"
    )
    parser.add_argument("--mode", choices=["train", "infer"], required=True)
    parser.add_argument("--reference", default=None)
    parser.add_argument("--reads",     required=True)
    parser.add_argument("--model",     default="output/model.pt")
    parser.add_argument("--output",    default="output/contigs.fasta")
    parser.add_argument("--k",              type=int,   default=21)
    parser.add_argument("--epochs",         type=int,   default=100)
    parser.add_argument("--lr",             type=float, default=1e-3)
    parser.add_argument("--hidden",         type=int,   default=64)
    parser.add_argument("--threshold",      type=float, default=0.5)
    parser.add_argument("--min-kmer-count", type=int,   default=3,
                        help="Remove k-mers appearing fewer than this many times (default 3)")
    args = parser.parse_args()

    if args.mode == "train":
        if not args.reference:
            parser.error("--reference is required for training")
        train(
            reference_fasta  = args.reference,
            reads_fastq      = args.reads,
            model_path       = args.model,
            k                = args.k,
            hidden_dim       = args.hidden,
            epochs           = args.epochs,
            lr               = args.lr,
            threshold        = args.threshold,
            min_kmer_count   = args.min_kmer_count,
        )
    elif args.mode == "infer":
        infer(
            reads_fastq  = args.reads,
            model_path   = args.model,
            output_fasta = args.output,
            threshold    = args.threshold,
        )


if __name__ == "__main__":
    main()
