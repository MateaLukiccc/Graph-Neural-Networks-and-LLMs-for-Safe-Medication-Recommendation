import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter


class MaskLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

    def forward(self, input, mask=None):
        if mask is None:
            masked_weight = self.weight
        else:
            mask = mask.to(device=self.weight.device, dtype=self.weight.dtype)
            if mask.shape == (self.out_features, self.in_features):
                use_mask = mask
            elif mask.shape == (self.in_features, self.out_features):
                use_mask = mask.t()
            else:
                raise RuntimeError(
                    f"Mask shape {tuple(mask.shape)} incompatible with expected "
                    f"(out_features, in_features)=({self.out_features},{self.in_features})"
                )
            masked_weight = self.weight * use_mask

        return F.linear(input, masked_weight, self.bias)

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.in_features} -> {self.out_features})"


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprint, dim, layer_hidden, device=torch.device("cpu")):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.device = device
        self.embed_fingerprint = nn.Embedding(N_fingerprint, dim)
        self.W_fingerprint = nn.ModuleList([nn.Linear(dim, dim) for _ in range(layer_hidden)])
        self.layer_hidden = layer_hidden

        self.attn_query = nn.Parameter(torch.empty(dim))
        init.xavier_uniform_(self.embed_fingerprint.weight)
        for lin in self.W_fingerprint:
            init.kaiming_uniform_(lin.weight, nonlinearity="relu")
            if lin.bias is not None:
                init.zeros_(lin.bias)

    def pad(self, matrices, pad_value):
        """Pad the list of matrices for batch processing into a single block-diagonal-like matrix."""
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        pad_matrices = torch.full(
            (M, N),
            fill_value=pad_value,
            device=self.device,
            dtype=matrices[0].dtype,
        )
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            m, n = shapes[k]
            pad_matrices[i : i + m, j : j + n] = matrix.to(self.device)
            i += m
            j += n
        return pad_matrices

    def attention_pooling(self, vectors, sizes):
        attn_q = self.attn_query.to(vectors.device)
        pooled = []
        for v in torch.split(vectors, sizes):
            # v: (num_atoms_i, dim)
            scores = v.matmul(attn_q)                # (num_atoms_i,)
            weights = torch.softmax(scores, dim=0)   # (num_atoms_i,)
            weighted_sum = (weights.unsqueeze(1) * v).sum(dim=0)
            pooled.append(weighted_sum)
        return torch.stack(pooled, dim=0)
    
    def update(self, matrix, vectors, layer):
        hidden_vectors = F.relu(self.W_fingerprint[layer](vectors))
        return hidden_vectors + torch.mm(matrix, hidden_vectors)

    def mean(self, vectors, axis):
        mean_vectors = [torch.mean(v, 0) for v in torch.split(vectors, axis)]
        return torch.stack(mean_vectors)

    def forward(self, inputs):
        fingerprints, adjacencies, molecular_sizes = inputs
        fingerprints = torch.cat(fingerprints).to(self.device)
        adjacencies = self.pad(adjacencies, 0)

        fingerprint_vectors = self.embed_fingerprint(fingerprints)
        for layer in range(self.layer_hidden):
            hs = self.update(adjacencies, fingerprint_vectors, layer)
            fingerprint_vectors = hs

        molecular_vectors = self.attention_pooling(fingerprint_vectors, molecular_sizes)
        return molecular_vectors

class SafeDrugModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        ddi_adj,
        ddi_mask_H,
        MPNNSet,
        N_fingerprints,
        average_projection,
        emb_dim=256,
        device=torch.device("cpu"),
    ):
        super(SafeDrugModel, self).__init__()
        self.device = device
        self.emb_dim = emb_dim

        # pre-embedding (two vocab types)
        self.embeddings = nn.ModuleList([nn.Embedding(vocab_size[i], emb_dim) for i in range(2)])
        self.dropout = nn.Dropout(p=0.5)

        # BIDIRECTIONAL GRUs (batch_first=True)
        self.encoders = nn.ModuleList([
            nn.GRU(emb_dim, emb_dim, batch_first=True, bidirectional=True)
            for _ in range(2)
        ])

        # After concatenating two bi-GRU outputs we get last-dim = 4 * emb_dim
        self.query = nn.Sequential(nn.ReLU(), nn.Linear(4 * emb_dim, emb_dim))

        # bipartite transform 
        self.bipartite_transform = nn.Sequential(nn.Linear(emb_dim, ddi_mask_H.shape[1]))
        self.bipartite_output = MaskLinear(ddi_mask_H.shape[1], vocab_size[2], bias=False)
        
        # Multi-Head Attention layer
        self.mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=4, batch_first=True)
        
        # Build MPNN, compute fixed embeddings and store as buffer (non-trainable constant)
        self.MPNN_molecule_Set = list(zip(*MPNNSet))
        mpnn = MolecularGraphNeuralNetwork(N_fingerprints, emb_dim, layer_hidden=2, device=device)
        with torch.no_grad():
            mpnn_emb = mpnn(self.MPNN_molecule_Set)  
            mpnn_emb = torch.mm(average_projection.to(device=device), mpnn_emb.to(device=device))

        self.register_buffer("MPNN_emb", mpnn_emb) 
        self.MPNN_output = nn.Linear(2 * emb_dim, vocab_size[2])
        self.MPNN_layernorm = nn.LayerNorm(vocab_size[2])

        # graphs, bipartite matrix
        self.register_buffer("tensor_ddi_adj", torch.FloatTensor(ddi_adj))
        self.register_buffer("tensor_ddi_mask_H", torch.FloatTensor(ddi_mask_H))
        self.init_weights()

    def forward(self, input):
        i1_seq = []
        i2_seq = []

        def sum_embedding(embedding):
            return embedding.sum(dim=1).unsqueeze(dim=0)

        for adm in input:
            a0 = torch.LongTensor(adm[0]).unsqueeze(dim=0).to(self.device)
            a1 = torch.LongTensor(adm[1]).unsqueeze(dim=0).to(self.device)
            i1 = sum_embedding(self.dropout(self.embeddings[0](a0)))
            i2 = sum_embedding(self.dropout(self.embeddings[1](a1)))
            i1_seq.append(i1)
            i2_seq.append(i2)

        i1_seq = torch.cat(i1_seq, dim=1)  # (1, seq, dim)
        i2_seq = torch.cat(i2_seq, dim=1)  # (1, seq, dim)

        # run bi-GRUs -> outputs shape (1, seq, emb_dim * 2)
        o1, h1 = self.encoders[0](i1_seq)
        o2, h2 = self.encoders[1](i2_seq)

        # concatenate along feature dim: (1, seq, emb_dim*4) then remove batch dim
        patient_representations = torch.cat([o1, o2], dim=-1).squeeze(dim=0)  # (seq, 4*emb_dim)

        # project to query vector (we use last visit)
        query = self.query(patient_representations)[-1:, :]  # (1, emb_dim)

        # ---------- Multi-Head Attention over molecule embeddings ----------
        # MPNN_emb: (num_molecules, emb_dim) -> (1, num_molecules, emb_dim)
        memory = self.MPNN_emb.unsqueeze(0)  # (1, num_molecules, emb_dim)
        query_mha = query.unsqueeze(0)  # (1, 1, emb_dim)

        # Multi-head attention: returns (1, 1, emb_dim)
        attn_output, _ = self.mha(query=query_mha, key=memory, value=memory)
        context = attn_output.squeeze(0)  # (1, emb_dim)

        # concat query + context -> project into vocab logits
        mpnn_concat = torch.cat([query, context], dim=-1)  # (1, 2*emb_dim)
        MPNN_out_logits = self.MPNN_output(mpnn_concat)  # (1, vocab_size[2])
        MPNN_att = self.MPNN_layernorm(MPNN_out_logits)

        # local embedding via bipartite transform + masked linear
        bip_input = torch.sigmoid(self.bipartite_transform(query))  # (1, ddi_mask_H.shape[1])
        bipartite_emb = self.bipartite_output(bip_input, self.tensor_ddi_mask_H.t())  # (1, vocab_size[2])

        result = torch.mul(bipartite_emb, MPNN_att)  # (1, vocab_size[2])

        neg_pred_prob = torch.sigmoid(result)
        neg_pred_prob = neg_pred_prob.t() * neg_pred_prob  # (vocab_size, vocab_size)

        batch_neg = 0.0005 * neg_pred_prob.mul(self.tensor_ddi_adj).sum()

        return result, batch_neg

    def init_weights(self):
        for emb in self.embeddings:
            init.xavier_uniform_(emb.weight)

        for gru in self.encoders:
            for name, param in gru.named_parameters():
                if "weight_ih" in name:
                    init.kaiming_uniform_(param, nonlinearity="relu")
                elif "weight_hh" in name:
                    init.orthogonal_(param)
                elif "bias" in name:
                    init.zeros_(param)

        for m in self.query:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

        for m in self.bipartite_transform:
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    init.zeros_(m.bias)

        if isinstance(self.bipartite_output, MaskLinear):
            init.xavier_uniform_(self.bipartite_output.weight)
            if self.bipartite_output.bias is not None:
                init.zeros_(self.bipartite_output.bias)

        init.kaiming_uniform_(self.MPNN_output.weight, nonlinearity="relu")
        if self.MPNN_output.bias is not None:
            init.zeros_(self.MPNN_output.bias)
