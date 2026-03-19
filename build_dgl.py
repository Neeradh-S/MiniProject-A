#!/usr/bin/env python3
import dgl
import numpy as np
import torch
from Bio import SeqIO
from collections import defaultdict, Counter
import os

# Create output folder
os.makedirs("output", exist_ok=True)

# ============================================================
# 1. BUILD DE BRUIJN GRAPH
# ============================================================
class DeBruijnGraph:
    def __init__(self, k=31):
        self.k = k
        self.nodes = set()
        self.edges = defaultdict(int)

    def add_sequence(self, seq):
        for i in range(len(seq) - self.k):
            k1 = seq[i:i + self.k]
            k2 = seq[i + 1:i + self.k + 1]
            self.nodes.add(k1)
            self.nodes.add(k2)
            self.edges[(k1, k2)] += 1

    @classmethod
    def from_fasta(cls, fasta_path, k=31):
        g = cls(k=k)
        for record in SeqIO.parse(fasta_path, "fasta"):
            g.add_sequence(str(record.seq))
        return g

    def save(self):
        """Save the raw DBG before mapping reads."""
        with open("output/dbg_nodes.txt", "w") as f:
            for n in self.nodes:
                f.write(n + "\n")

        with open("output/dbg_edges.txt", "w") as f:
            for (u, v), count in self.edges.items():
                f.write(f"{u}\t{v}\t{count}\n")


# ============================================================
# 2. MAP FASTQ READS
# ============================================================
class ReadMapper:
    def __init__(self, dbg: DeBruijnGraph):
        self.dbg = dbg
        self.k = dbg.k
        self.node_counts = Counter()
        self.edge_counts = Counter()

    def map_read(self, read):
        seq = str(read.seq)
        for i in range(len(seq) - self.k):
            k1 = seq[i:i + self.k]
            k2 = seq[i + 1:i + self.k + 1]

            if k1 in self.dbg.nodes:
                self.node_counts[k1] += 1

            if (k1, k2) in self.dbg.edges:
                self.edge_counts[(k1, k2)] += 1

    @classmethod
    def from_fastq(cls, dbg, fastq_path):
        mapper = cls(dbg)
        for record in SeqIO.parse(fastq_path, "fastq"):
            mapper.map_read(record)
        return mapper


# ============================================================
# 3. GRAPH ANNOTATION
# ============================================================
class GraphAnnotator:
    def __init__(self, dbg: DeBruijnGraph, mapper: ReadMapper):
        self.dbg = dbg
        self.mapper = mapper

    def compute_node_features(self):
        counts = np.array([self.mapper.node_counts[n] for n in self.dbg.nodes])
        mean = counts.mean() if len(counts) > 0 else 0
        std = counts.std() if len(counts) > 0 else 1

        zscores = {n: (self.mapper.node_counts[n] - mean) / (std + 1e-9)
                   for n in self.dbg.nodes}

        # Save node coverage and z scores
        with open("output/node_coverage.txt", "w") as f:
            for n in self.dbg.nodes:
                f.write(f"{n}\t{self.mapper.node_counts[n]}\n")

        with open("output/node_zscores.txt", "w") as f:
            for n in self.dbg.nodes:
                f.write(f"{n}\t{zscores[n]}\n")

        return zscores

    def compute_edge_features(self):
        edges = list(self.dbg.edges.keys())
        counts = np.array([self.mapper.edge_counts[e] for e in edges])
        mean = counts.mean() if len(counts) > 0 else 0
        std = counts.std() if len(counts) > 0 else 1

        z = {e: (self.mapper.edge_counts[e] - mean) / (std + 1e-9)
             for e in edges}

        # Save edge z scores
        with open("output/edge_zscores.txt", "w") as f:
            for (u, v) in edges:
                f.write(f"{u}\t{v}\t{z[(u, v)]}\n")

        return z


# ============================================================
# 4. BUILD DGL GRAPH
# ============================================================
class DGLGraphBuilder:
    def __init__(self, dbg: DeBruijnGraph, node_z, edge_z, mapper):
        self.dbg = dbg
        self.node_z =   node_z
        self.edge_z = edge_z
        self.mapper = mapper

    def build(self):
        nodes = list(self.dbg.nodes)
        node_index = {n: i for i, n in enumerate(nodes)}

        src, dst, edge_feats = [], [], []

        for (u, v) in self.dbg.edges.keys():
            src.append(node_index[u])
            dst.append(node_index[v])
            edge_feats.append(self.edge_z[(u, v)])

        g = dgl.graph((src, dst), device="cpu")

        g.ndata["z"] = torch.tensor([self.node_z[n] for n in nodes], dtype=torch.float32)
        g.ndata["coverage"] = torch.tensor([self.mapper.node_counts[n] for n in nodes], dtype=torch.float32)
        g.edata["z"] = torch.tensor(edge_feats, dtype=torch.float32)

        # Save nodes
        with open("output/nodes.txt", "w") as f:
            for n in nodes:
                f.write(n + "\n")

        # Save edges
        with open("output/edges.txt", "w") as f:
            for (u, v) in self.dbg.edges.keys():
                f.write(f"{u}\t{v}\n")

        # Save DGL graph
        dgl.save_graphs("output/graph.bin", [g])

        return g, nodes, list(self.dbg.edges.keys())


# ============================================================
# 5. PIPELINE
# ============================================================
def build_all(reference_fasta, reads_fastq, k=31):
    print("Building De Bruijn graph…")
    dbg = DeBruijnGraph.from_fasta(reference_fasta, k)
    dbg.save()

    print("Mapping FASTQ reads…")
    mapper = ReadMapper.from_fastq(dbg, reads_fastq)

    print("Computing features…")
    annot = GraphAnnotator(dbg, mapper)
    node_z = annot.compute_node_features()
    edge_z = annot.compute_edge_features()

    print("Building DGL graph…")
    builder = DGLGraphBuilder(dbg, node_z, edge_z, mapper)
    g, nodes, edges = builder.build()

    return g, nodes, edges


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == "__main__":
    g, nodes, edges = build_all(
        "./curly-octo-train/reference.fasta",
        "./curly-octo-train/hifi_sim_small_0001.fq",
        k=31
    )
    print("Done. Graph has:")
    print("Nodes:", len(nodes))
    print("Edges:", len(edges))
    print("Saved all outputs to ./output/")
