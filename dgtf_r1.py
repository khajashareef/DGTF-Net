"""
DGTF-Net -- revision (R1) experiment module for Discover Computing.

Everything that produces a number in the revised manuscript lives here:
  * data loading / windowing / metrics       (identical to the original submission notebook)
  * DGTF-Net with ablation switches          (default switches == original model)
  * wrappers that run the OFFICIAL code of STAEformer, STID, AGCRN, D2STGNN and MegaCRN
    inside the same training loop (same split, normalisation, loss, optimiser, schedule, epochs)
  * a resumable multi-seed job queue that writes one JSON (+ one NPZ) per run to Google Drive
  * fuzzy congestion-severity evaluation (Sec. 4.8 of the revised manuscript)
  * aggregation: mean +/- std tables, Welch / Mann-Whitney tests across seeds,
    Diebold-Mariano tests on per-window errors, paired tests for the ablation.

STGCN / DCRNN / Graph WaveNet use the BasicTS benchmark implementations (pinned commit), run in the same loop.
"""
import os, sys, json, time, math, csv, random, types, importlib.util, subprocess
import numpy as np
import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------
# 0. Global protocol (unchanged from the submitted manuscript)
# ----------------------------------------------------------------------------------------
P, Q = 12, 12
SPD = 288
D_MODEL, L_BLOCKS, HEADS, DE, TOPK = 64, 3, 8, 10, 20
BATCH, EPOCHS, PATIENCE, LR = 32, 30, 12, 1e-3
WEIGHT_DECAY, CLIP, HUBER_DELTA = 1e-5, 5.0, 1.0
TRAIN_FRAC, VAL_FRAC = 0.6, 0.2                       # 60/20/20 chronological split
HORIZONS = (3, 6, 12)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

EXT_DIR = os.environ.get('DGTF_EXT', '/content/ext')    # where official repos are cloned
DATA_DIR = os.environ.get('DGTF_DATA', '/content/data')

OFFICIAL_REPOS = {   # pinned commits, recorded 2026-11-08
    'STAEformer': ('https://github.com/XDZhelheim/STAEformer.git', 'fc49d39b2f1a8e3cf37b6289d7240680e1690f3f'),
    'STID':       ('https://github.com/zezhishao/STID.git',        'e8b313bc591bdd0101a1619962c9b503e75127c0'),
    'AGCRN':      ('https://github.com/LeiBAI/AGCRN.git',          '7fbbf2aeb099242098a3cf482b55cd45d7295c28'),
    'D2STGNN':    ('https://github.com/zezhishao/D2STGNN.git',     '82c2d38ba098a6aa87e8099c1ee8da58480692ec'),
    'MegaCRN':    ('https://github.com/deepkashiwa20/MegaCRN.git', '5158190948de7e8628f84780a429eb9189f5908f'),
    # BasicTS benchmark (Shao et al., TKDE 2024), last commit that still ships STGCN / DCRNN / GWNet
    'BasicTS':    ('https://github.com/GestaltCogTeam/BasicTS.git', 'c218c07b6ce5e4cf908b147fd180c486346fed9c'),
}
BASICTS_DIRS = ['baselines/STGCN/arch', 'baselines/DCRNN/arch', 'baselines/GWNet/arch']


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False


def gpu_name():
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return 'cpu'


# ----------------------------------------------------------------------------------------
# 1. Data (identical to submission: flow channel, Gaussian-kernel prior graph, row-normalised)
# ----------------------------------------------------------------------------------------
def _fetch_pems(name):
    os.makedirs(DATA_DIR, exist_ok=True)
    npz, csvf = os.path.join(DATA_DIR, f'{name}.npz'), os.path.join(DATA_DIR, f'{name}.csv')
    if not os.path.exists(npz):
        tmp = os.path.join(DATA_DIR, f'_ast_{name}')
        os.system(f'git clone --depth 1 --filter=blob:none --sparse https://github.com/guoshnBJTU/ASTGNN.git {tmp} -q')
        os.system(f'cd {tmp} && git sparse-checkout set data/{name} -q')
        os.system(f'cp {tmp}/data/{name}/{name}.npz {tmp}/data/{name}/{name}.csv {DATA_DIR}/')
    return npz, csvf


def load_dataset(name):
    npz, csvf = _fetch_pems(name)
    arr = np.load(npz)['data'][:, :, 0:1].astype('float32')          # (T, N, 1) flow
    N = arr.shape[1]
    A = np.zeros((N, N), 'float32')
    rows = list(csv.reader(open(csvf)))[1:]
    ds = [(int(a), int(b), float(c)) for a, b, c in rows]
    sg = np.std([c for *_, c in ds]) or 1.0
    for a, b, c in ds:
        w = np.exp(-(c ** 2) / (2 * sg ** 2 + 1e-8)); A[a, b] = w; A[b, a] = w
    np.fill_diagonal(A, 1.0)
    return arr, A / A.sum(1, keepdims=True), dict(name=name, N=N, channel='flow')


def build_windows(series):
    T, N, _ = series.shape
    n_tr, n_va = int(T * TRAIN_FRAC), int(T * VAL_FRAC)
    mean, std = series[:n_tr].mean(), series[:n_tr].std()               # global z-score (train only)
    norm = (series - mean) / std
    tod = (np.arange(T) % SPD).astype('int64'); dow = ((np.arange(T) // SPD) % 7).astype('int64')

    def win(lo, hi):
        xs, ys, td, dw = [], [], [], []
        for s in range(lo, hi - P - Q + 1):
            xs.append(norm[s:s + P]); ys.append(series[s + P:s + P + Q, :, 0])
            td.append(tod[s:s + P]); dw.append(dow[s:s + P])
        t = lambda a: torch.tensor(np.stack(a))
        return t(xs), t(ys), t(td), t(dw)
    return win(0, n_tr), win(n_tr, n_tr + n_va), win(n_tr + n_va, T), float(mean), float(std), N


def metrics(pred, true, horizons=None):
    """Masked MAE/RMSE/MAPE. Entries with true <= 1e-3 are excluded from ALL three metrics.
    'avg' is computed over ALL Q=12 output steps (not only 3/6/12)."""
    horizons = horizons or list(range(1, Q + 1))
    out = {}
    for h in horizons:
        ph, th = pred[:, h - 1], true[:, h - 1]; m = th > 1e-3
        out[h] = dict(MAE=float(np.abs(ph - th)[m].mean()),
                      RMSE=float(np.sqrt(((ph - th) ** 2)[m].mean())),
                      MAPE=float((np.abs(ph - th)[m] / th[m]).mean() * 100))
    m = true > 1e-3
    out['avg'] = dict(MAE=float(np.abs(pred - true)[m].mean()),
                      RMSE=float(np.sqrt(((pred - true) ** 2)[m].mean())),
                      MAPE=float((np.abs(pred - true)[m] / true[m]).mean() * 100))
    return out


def per_window_errors(pred, true):
    """(W, Q) masked MAE per test window and step, averaged over nodes -> for DM tests."""
    m = (true > 1e-3).astype('float32')
    e = np.abs(pred - true) * m
    return (e.sum(2) / np.maximum(m.sum(2), 1)).astype('float32')


# ----------------------------------------------------------------------------------------
# 2. DGTF-Net with ablation switches. Default arguments reproduce the submitted model exactly.
# ----------------------------------------------------------------------------------------
class DGTFNet(nn.Module):
    """
    adaptive : learn A_adp and fuse with prior graph via scalar gate g        (Eq. 4-5)
    prior    : keep the prior distance graph in the fusion (False -> A~ = A_adp, g fixed to 1)
    spatial  : 'attn' (graph-biased multi-head attention, Eq. 6-8) | 'gcn' (top-k graph convolution)
    temporal : Transformer temporal branch                                     (Eq. 9-11)
    gate     : gated fusion (Eq. 12-13); False -> plain sum of the two branches
    embed    : time-of-day / day-of-week / node embeddings in Eq. 3
    gate_mode: 'static' (one learned scalar g, used in all reported results) | 'time'
               (g_t = sigmoid(w^T pool(Z_t)), one gate per input window)
    """

    def __init__(self, N, A_pre, d=D_MODEL, L=L_BLOCKS, heads=HEADS, de=DE, topk=TOPK, P=P, Q=Q,
                 adaptive=True, temporal=True, gate=True, prior=True, spatial='attn', embed=True,
                 gate_mode='static'):
        super().__init__()
        self.N, self.d, self.h, self.topk, self.P = N, d, heads, min(topk, N), P
        self.adaptive, self.temporal, self.gate = adaptive, temporal, gate
        self.prior, self.spatial, self.embed, self.gate_mode = prior, spatial, embed, gate_mode
        self.register_buffer('A_pre', A_pre)
        self.Win = nn.Linear(1, d); self.tod = nn.Embedding(SPD, d); self.dow = nn.Embedding(7, d)
        self.node = nn.Parameter(torch.randn(N, d) * 0.02)
        if adaptive:
            self.E1 = nn.Parameter(torch.randn(N, de) * 0.1); self.E2 = nn.Parameter(torch.randn(N, de) * 0.1)
            self.gate_logit = nn.Parameter(torch.zeros(()))
        self.blocks = nn.ModuleList()
        for _ in range(L):
            b = nn.ModuleDict({'sp': nn.MultiheadAttention(d, heads, batch_first=True)})
            if temporal:
                b['tp'] = nn.MultiheadAttention(d, heads, batch_first=True)
                b['ln1'] = nn.LayerNorm(d); b['ln2'] = nn.LayerNorm(d)
                b['ff'] = nn.Sequential(nn.Linear(d, 4 * d), nn.ReLU(), nn.Linear(4 * d, d))
            if gate:
                b['gs'] = nn.Linear(d, d); b['gt'] = nn.Linear(d, d)
            b['lnb'] = nn.LayerNorm(d); self.blocks.append(b)
        self.head = nn.Sequential(nn.Linear(P * d, 2 * d), nn.ReLU(), nn.Linear(2 * d, Q))
        # extra parameters only for non-default variants (created last so the default model's
        # parameter initialisation order is unchanged)
        if spatial == 'gcn':
            self.gcn = nn.ModuleList([nn.Linear(d, d) for _ in range(L)])
        if adaptive and gate_mode == 'time':
            self.wg = nn.Linear(d, 1)
            nn.init.zeros_(self.wg.weight); nn.init.zeros_(self.wg.bias)   # starts at g = 0.5
        self._last_g = None

    # --- graph ---------------------------------------------------------------------------
    def _adp(self):
        return torch.softmax(torch.relu(self.E1 @ self.E2.t()), 1)

    def gate_value(self):
        if not self.adaptive:
            return 0.0
        if not self.prior:
            return 1.0
        return float(torch.sigmoid(self.gate_logit).item())

    def _graph(self, z=None):
        if not self.adaptive:
            return self.A_pre
        A_adp = self._adp()
        if not self.prior:
            return A_adp
        if self.gate_mode == 'time':
            g = torch.sigmoid(self.wg(z.mean(dim=(1, 2))))[:, :, None]      # (B,1,1)
            self._last_g = g.detach().flatten()
            return g * A_adp[None] + (1 - g) * self.A_pre[None]              # (B,N,N)
        g = torch.sigmoid(self.gate_logit)
        return g * A_adp + (1 - g) * self.A_pre

    def _bias(self, A):
        """Top-k sparsification of A~ (row-wise, k = TOPK) -> additive attention bias log A~_ij.
        Entries outside the k strongest neighbours of node i get -inf (masked out)."""
        kth = torch.topk(A, self.topk, -1).values[..., -1:]
        return torch.log(A + 1e-6) + torch.where(A >= kth, 0.0, float('-inf'))

    def _sparse_norm(self, A):          # for the 'gcn' variant: same top-k support, row-normalised
        kth = torch.topk(A, self.topk, -1).values[..., -1:]
        As = torch.where(A >= kth, A, torch.zeros_like(A))
        return As / (As.sum(-1, keepdim=True) + 1e-8)

    # --- forward -------------------------------------------------------------------------
    def forward(self, x, td, dw):
        B, P = x.shape[0], self.P
        z = self.Win(x)
        if self.embed:
            z = z + self.tod(td)[:, :, None, :] + self.dow(dw)[:, :, None, :] + self.node[None, None]
        A = self._graph(z)
        if self.spatial == 'attn':
            bias = self._bias(A)
            if bias.dim() == 3:                                              # time-conditioned gate
                bias = bias[:, None, None].expand(B, P, self.h, self.N, self.N).reshape(B * P * self.h, self.N, self.N)
        else:
            As = self._sparse_norm(A)
        for li, b in enumerate(self.blocks):
            if self.spatial == 'attn':
                si = z.reshape(B * P, self.N, self.d)
                hs, _ = b['sp'](si, si, si, attn_mask=bias); hs = hs.reshape(B, P, self.N, self.d)
            else:
                agg = torch.einsum('nm,bpmd->bpnd', As, z) if As.dim() == 2 else torch.einsum('bnm,bpmd->bpnd', As, z)
                hs = torch.relu(self.gcn[li](agg))
            if self.temporal:
                ti = z.permute(0, 2, 1, 3).reshape(B * self.N, P, self.d)
                ht, _ = b['tp'](ti, ti, ti); ht = b['ln1'](ti + ht); ht = b['ln2'](ht + b['ff'](ht))
                ht = ht.reshape(B, self.N, P, self.d).permute(0, 2, 1, 3)
            else:
                ht = z            # no temporal branch: the block input is passed through unchanged
            if self.gate:
                G = torch.sigmoid(b['gs'](hs) + b['gt'](ht)); hf = G * hs + (1 - G) * ht
            else:
                hf = hs + ht
            z = b['lnb'](z + hf)
        y = self.head(z.permute(0, 2, 1, 3).reshape(B, self.N, P * self.d))
        return y.permute(0, 2, 1), None


class GRUBaseline(nn.Module):
    def __init__(self, hidden=64, Q=Q):
        super().__init__(); self.rnn = nn.GRU(1, hidden, 2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(hidden, Q)

    def forward(self, x, td, dw):
        B, Pw, N, C = x.shape; o, _ = self.rnn(x.permute(0, 2, 1, 3).reshape(B * N, Pw, C))
        return self.head(o[:, -1]).reshape(B, N, -1).permute(0, 2, 1), None


# ----------------------------------------------------------------------------------------
# 3. Official implementations of the newer baselines, run inside the same loop
#    Interface for every model: forward(x, td, dw) -> (y_norm (B,Q,N), aux_loss or None)
# ----------------------------------------------------------------------------------------
def clone_official(names=None):
    os.makedirs(EXT_DIR, exist_ok=True)
    for name in names or OFFICIAL_REPOS:
        url, commit = OFFICIAL_REPOS[name]
        path = os.path.join(EXT_DIR, name)
        if os.path.exists(path):
            continue
        if name == 'BasicTS':                                   # sparse, blob-less clone of three folders
            subprocess.run(['git', 'clone', '-q', '--filter=blob:none', '--no-checkout', url, path], check=True)
            subprocess.run(['git', '-C', path, 'sparse-checkout', 'set'] + BASICTS_DIRS, check=True)
            subprocess.run(['git', '-C', path, 'checkout', '-q', commit], check=True)
        else:
            subprocess.run(['git', 'clone', '-q', url, path], check=True)
            subprocess.run(['git', '-C', path, 'checkout', '-q', commit], check=True)


def _load_file(modname, path):
    spec = importlib.util.spec_from_file_location(modname, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[modname] = mod; spec.loader.exec_module(mod)
    return mod


def _tod_frac(td):
    return td.float() / SPD


class STAEformerWrap(nn.Module):
    """Official STAEformer (CIKM'23), PEMS04/08 hyper-parameters from the official STAEformer.yaml."""
    def __init__(self, N, A_pre=None):
        super().__init__()
        try:
            import torchinfo  # noqa: F401  (imported at top of the official file)
        except ImportError:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'torchinfo'], check=True)
        mod = _load_file('staeformer_official', os.path.join(EXT_DIR, 'STAEformer/model/STAEformer.py'))
        self.m = mod.STAEformer(num_nodes=N, in_steps=P, out_steps=Q, steps_per_day=SPD, input_dim=3, output_dim=1,
                                input_embedding_dim=24, tod_embedding_dim=24, dow_embedding_dim=24,
                                spatial_embedding_dim=0, adaptive_embedding_dim=80, feed_forward_dim=256,
                                num_heads=4, num_layers=3, dropout=0.1)

    def forward(self, x, td, dw):
        N = x.shape[2]
        tod = _tod_frac(td)[:, :, None, None].expand(-1, -1, N, 1)
        dow = dw.float()[:, :, None, None].expand(-1, -1, N, 1)
        out = self.m(torch.cat([x, tod, dow], -1))              # (B, Q, N, 1)
        return out[..., 0], None


class STIDWrap(nn.Module):
    """Official STID (CIKM'22), PEMS04/08 hyper-parameters from stid/PEMS0X.py."""
    def __init__(self, N, A_pre=None):
        super().__init__()
        sys.path.insert(0, os.path.join(EXT_DIR, 'STID'))
        mlp = _load_file('stid_official.mlp', os.path.join(EXT_DIR, 'STID/stid/arch/mlp.py'))
        sys.modules['stid_official'] = types.ModuleType('stid_official'); sys.modules['stid_official'].__path__ = []
        src = open(os.path.join(EXT_DIR, 'STID/stid/arch/stid_arch.py')).read().replace('from .mlp import', 'from stid_official.mlp import')
        mod = types.ModuleType('stid_official.arch'); exec(compile(src, 'stid_arch.py', 'exec'), mod.__dict__)
        self.m = mod.STID(num_nodes=N, input_len=P, input_dim=3, embed_dim=32, output_len=Q, num_layer=3,
                          if_node=True, node_dim=32, if_T_i_D=True, if_D_i_W=True, temp_dim_tid=32,
                          temp_dim_diw=32, time_of_day_size=SPD, day_of_week_size=7)

    def forward(self, x, td, dw):
        N = x.shape[2]
        tod = _tod_frac(td)[:, :, None, None].expand(-1, -1, N, 1)
        dow = (dw.float() / 7)[:, :, None, None].expand(-1, -1, N, 1)
        out = self.m(torch.cat([x, tod, dow], -1), None, 0, 0, self.training)   # (B, Q, N, 1)
        return out[..., 0], None


class AGCRNWrap(nn.Module):
    """Official AGCRN (NeurIPS'20); PEMSD4/8 .conf hyper-parameters; official parameter init."""
    def __init__(self, N, A_pre=None, embed_dim=None):
        super().__init__()
        cell = os.path.join(EXT_DIR, 'AGCRN/model')
        if 'model' not in sys.modules:
            sys.modules['model'] = types.ModuleType('model'); sys.modules['model'].__path__ = []
        agcn = _load_file('agcrn_AGCN', os.path.join(cell, 'AGCN.py'))
        sys.modules['model.AGCN'] = agcn
        src = open(os.path.join(cell, 'AGCRNCell.py')).read()
        c = types.ModuleType('model.AGCRNCell'); exec(compile(src, 'AGCRNCell.py', 'exec'), c.__dict__)
        sys.modules['model.AGCRNCell'] = c
        src = open(os.path.join(cell, 'AGCRN.py')).read()
        m = types.ModuleType('agcrn_official'); exec(compile(src, 'AGCRN.py', 'exec'), m.__dict__)
        args = types.SimpleNamespace(num_nodes=N, input_dim=1, rnn_units=64, output_dim=1, horizon=Q,
                                     num_layers=2, default_graph=True, cheb_k=2,
                                     embed_dim=embed_dim or (10 if N > 200 else 2))   # PEMSD4: 10, PEMSD8: 2
        self.m = m.AGCRN(args)
        for p in self.m.parameters():                                          # as in official Run.py
            nn.init.xavier_uniform_(p) if p.dim() > 1 else nn.init.uniform_(p)

    def forward(self, x, td, dw):
        return self.m(x, None)[..., 0], None


class D2STGNNWrap(nn.Module):
    """Official D2STGNN (VLDB'22); PEMS04/08 yaml hyper-parameters; prior graph as double transition."""
    def __init__(self, N, A_pre):
        super().__init__()
        root = os.path.join(EXT_DIR, 'D2STGNN')
        if root not in sys.path:
            sys.path.insert(0, root)
        pkg = types.ModuleType('models'); pkg.__path__ = [os.path.join(root, 'models')]; sys.modules['models'] = pkg
        from models.model import D2STGNN
        A = A_pre.detach().cpu()
        self.m = D2STGNN(num_feat=1, num_hidden=32, node_hidden=10, time_emb_dim=10, dropout=0.1, seq_length=Q,
                         k_t=3, k_s=2, gap=3, num_modalities=2, num_nodes=N,
                         adjs=[A.T.contiguous(), A.T.contiguous()])
        self._adjs = self.m._model_args['adjs']

    def _apply(self, fn, *a, **k):                    # keep the prior-graph tensors on the model's device
        super()._apply(fn, *a, **k)
        for i, t in enumerate(self._adjs):
            self._adjs[i] = fn(t)
        return self

    def forward(self, x, td, dw):
        N = x.shape[2]
        tod = _tod_frac(td)[:, :, None, None].expand(-1, -1, N, 1)
        dow = dw.float()[:, :, None, None].expand(-1, -1, N, 1)
        out = self.m(torch.cat([x, tod, dow], -1))                # (B, N, Q)
        return out.transpose(1, 2), None


class MegaCRNWrap(nn.Module):
    """Official MegaCRN (AAAI'23) with its separation + compactness auxiliary losses (lambda = 0.01 each).
    Curriculum learning is disabled (free-running decoder), as for DCRNN, to stay in the shared loop."""
    def __init__(self, N, A_pre=None):
        super().__init__()
        mod = _load_file('megacrn_official', os.path.join(EXT_DIR, 'MegaCRN/model/MegaCRN.py'))
        self.m = mod.MegaCRN(num_nodes=N, input_dim=1, output_dim=1, horizon=Q, rnn_units=64, num_layers=1,
                             cheb_k=3, ycov_dim=1, mem_num=20, mem_dim=64, use_curriculum_learning=False)
        self.sep = nn.TripletMarginLoss(margin=1.0); self.lam1 = self.lam2 = 0.01

    def forward(self, x, td, dw):
        N = x.shape[2]
        fut = (td[:, -1:] + torch.arange(1, Q + 1, device=td.device)[None]) % SPD
        ycov = (fut.float() / SPD)[:, :, None, None].expand(-1, -1, N, 1)
        out, h_att, query, pos, neg = self.m(x, ycov)
        aux = None
        if self.training:
            aux = self.lam1 * self.sep(query, pos.detach(), neg.detach()) + \
                  self.lam2 * nn.functional.mse_loss(query, pos.detach())
        return out[..., 0], aux


def _load_pkg(pkgname, folder, module):
    """Import folder/module.py as pkgname.module without executing folder/__init__.py (relative imports work)."""
    if pkgname not in sys.modules:
        pkg = types.ModuleType(pkgname); pkg.__path__ = [folder]; sys.modules[pkgname] = pkg
    return importlib.import_module(f'{pkgname}.{module}')


def _sym_norm_adj(A_pre):
    """A_pre = D^-1 A with A symmetric  ->  sqrt(A_pre * A_pre^T) = D^-1/2 A D^-1/2 exactly."""
    return torch.sqrt(A_pre * A_pre.T)


class STGCNWrap(nn.Module):
    """STGCN (IJCAI'18), PyTorch implementation from BasicTS (ChebGraphConv variant, ref. hazdzz/STGCN);
    hyper-parameters of BasicTS baselines/STGCN/PEMS0X.py: Ks=Kt=3, blocks [[1],[64,16,64],[64,16,64],[128,128],[12]],
    GLU, dropout 0.5, graph shift operator = symmetric normalised Laplacian of the prior graph."""
    def __init__(self, N, A_pre):
        super().__init__()
        mod = _load_pkg('bts_stgcn', os.path.join(EXT_DIR, 'BasicTS/baselines/STGCN/arch'), 'stgcn_arch')
        gso = torch.eye(N) - _sym_norm_adj(A_pre.detach().cpu().float())
        self.m = mod.STGCNChebGraphConv(Kt=3, Ks=3, blocks=[[1], [64, 16, 64], [64, 16, 64], [128, 128], [Q]], T=P,
                                         n_vertex=N, act_func='glu', graph_conv_type='cheb_graph_conv', gso=gso,
                                         bias=True, droprate=0.5)

    def forward(self, x, td, dw):
        return self.m(x, None, 0, 0, self.training)[..., 0], None          # (B, Q, N)


class DCRNNWrap(nn.Module):
    """DCRNN (ICLR'18), PyTorch implementation from BasicTS (port of chnsh/DCRNN_PyTorch); hyper-parameters of
    BasicTS baselines/DCRNN/PEMS0X.py (2 layers x 64 units, diffusion step 2, inputs = flow + time of day,
    double-transition supports). Curriculum learning disabled (free-running decoder) to stay in the shared loop.
    The cell creates its weights lazily, so one dummy forward pass is run here, before the optimiser is built."""
    def __init__(self, N, A_pre):
        super().__init__()
        mod = _load_pkg('bts_dcrnn', os.path.join(EXT_DIR, 'BasicTS/baselines/DCRNN/arch'), 'dcrnn_arch')
        A = A_pre.detach().cpu().float()
        self.m = mod.DCRNN(adj_mx=[A.T.contiguous(), A.T.contiguous()], cl_decay_steps=2000, horizon=Q, input_dim=2,
                           max_diffusion_step=2, num_nodes=N, num_rnn_layers=2, output_dim=1, rnn_units=64, seq_len=P,
                           use_curriculum_learning=False, filter_type='dual_random_walk')
        with torch.no_grad():                                            # materialise lazily-created parameters
            self.m(torch.zeros(1, P, N, 2), None, batch_seen=1)

    def forward(self, x, td, dw):
        N = x.shape[2]
        tod = _tod_frac(td)[:, :, None, None].expand(-1, -1, N, 1)
        return self.m(torch.cat([x, tod], -1), None, batch_seen=1)[..., 0], None


class GWNetWrap(nn.Module):
    """Graph WaveNet (IJCAI'19), implementation from BasicTS (port of nnzhan/Graph-WaveNet); hyper-parameters of
    BasicTS baselines/GWNet/PEMS0X.py (adaptive adjacency + double-transition prior supports, 4 blocks x 2 layers,
    dropout 0.3, inputs = flow + time of day)."""
    def __init__(self, N, A_pre):
        super().__init__()
        mod = _load_pkg('bts_gwnet', os.path.join(EXT_DIR, 'BasicTS/baselines/GWNet/arch'), 'gwnet_arch')
        A = A_pre.detach().cpu().float()
        self.m = mod.GraphWaveNet(num_nodes=N, dropout=0.3, supports=[A.T.contiguous(), A.T.contiguous()],
                                  gcn_bool=True, addaptadj=True, aptinit=None, in_dim=2, out_dim=Q,
                                  residual_channels=32, dilation_channels=32, skip_channels=256, end_channels=512,
                                  kernel_size=2, blocks=4, layers=2)

    def forward(self, x, td, dw):
        N = x.shape[2]
        tod = _tod_frac(td)[:, :, None, None].expand(-1, -1, N, 1)
        return self.m(torch.cat([x, tod], -1), None, 0, 0, self.training)[..., 0], None


MODEL_REGISTRY = {
    'DGTF-Net': lambda N, A: DGTFNet(N, A),
    'GRU':      lambda N, A: GRUBaseline(),
    'STGCN':         lambda N, A: STGCNWrap(N, A),
    'DCRNN':         lambda N, A: DCRNNWrap(N, A),
    'Graph WaveNet': lambda N, A: GWNetWrap(N, A),
    'STAEformer': lambda N, A: STAEformerWrap(N, A),
    'STID':       lambda N, A: STIDWrap(N, A),
    'AGCRN':      lambda N, A: AGCRNWrap(N, A),
    'D2STGNN':    lambda N, A: D2STGNNWrap(N, A),
    'MegaCRN':    lambda N, A: MegaCRNWrap(N, A),
}

ABLATIONS = {
    'Full DGTF-Net':               dict(),
    'w/o adaptive graph':          dict(adaptive=False),
    'w/o prior graph':             dict(prior=False),
    'GCN instead of graph attn.':  dict(spatial='gcn'),
    'w/o Transformer':             dict(temporal=False),
    'w/o gated fusion':            dict(gate=False),
    'w/o adaptive embeddings':     dict(embed=False),
    'time-conditioned gate':       dict(gate_mode='time'),
}


def register_model(name, factory):
    """factory(N, A_pre_tensor) -> nn.Module with forward(x, td, dw) -> (y_norm (B,Q,N), aux or None)."""
    MODEL_REGISTRY[name] = factory


# ----------------------------------------------------------------------------------------
# 4. Fuzzy congestion-severity head (Sec. 4.8): volume-to-capacity ratio + triangular MFs
# ----------------------------------------------------------------------------------------
def fuzzy_setup(series, n_tr):
    """c_i  = 95th percentile of training flow at sensor i (reference capacity)
       r    = flow / c_i (volume-to-capacity ratio)
       b1,b2 = 50th / 85th percentiles of training r (pooled over sensors, zeros excluded)
       Triangular MFs form a Ruspini partition whose cross-over points are exactly b1 and b2."""
    tr = series[:n_tr, :, 0]
    cap = np.percentile(tr, 95, axis=0).astype('float32'); cap[cap <= 0] = 1.0
    r = (tr / cap); r = r[tr > 1e-3]
    b1, b2 = float(np.percentile(r, 50)), float(np.percentile(r, 85))
    cM = (b1 + b2) / 2; cL = 2 * b1 - cM; cH = 2 * b2 - cM
    return dict(cap=cap.tolist(), b1=b1, b2=b2, cL=cL, cM=cM, cH=cH)


def fuzzy_memberships(r, fz):
    cL, cM, cH = fz['cL'], fz['cM'], fz['cH']
    low = np.clip((cM - r) / (cM - cL), 0, 1)                        # left shoulder: 1 for r <= cL
    high = np.clip((r - cM) / (cH - cM), 0, 1)                       # right shoulder: 1 for r >= cH
    med = np.clip(np.minimum((r - cL) / (cM - cL), (cH - r) / (cH - cM)), 0, 1)
    return np.stack([low, med, high], -1)


def fuzzy_classify(flow, fz):
    """Mamdani rules R1-R3 (IF r is Low/Medium/High THEN severity Low/Medium/High),
    max-membership defuzzification. Returns labels 0/1/2 and membership degrees."""
    r = flow / np.asarray(fz['cap'], 'float32')[None, None, :]
    mu = fuzzy_memberships(r, fz)
    return mu.argmax(-1), mu


def crisp_labels(flow, fz):
    r = flow / np.asarray(fz['cap'], 'float32')[None, None, :]
    return np.digitize(r, [fz['b1'], fz['b2']])


def fuzzy_eval(pred, true, fz):
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
    yp, _ = fuzzy_classify(pred, fz); yt = crisp_labels(true, fz); m = true > 1e-3
    out = {}
    for h in list(HORIZONS) + ['all']:
        sl = slice(None) if h == 'all' else h - 1
        a, b = yt[:, sl][m[:, sl]].ravel(), yp[:, sl][m[:, sl]].ravel()
        out[str(h)] = dict(acc=float(accuracy_score(a, b)),
                           macro_P=float(precision_score(a, b, average='macro', zero_division=0)),
                           macro_R=float(recall_score(a, b, average='macro', zero_division=0)),
                           macro_F1=float(f1_score(a, b, average='macro', zero_division=0)),
                           F1_per_class=[float(v) for v in f1_score(a, b, average=None, labels=[0, 1, 2], zero_division=0)])
    return out


# ----------------------------------------------------------------------------------------
# 5. Training loop (identical to submission, plus: per-run seed, optional aux loss, logging)
# ----------------------------------------------------------------------------------------
def train_eval(model, name, data, fz=None, epochs=EPOCHS, log=print):
    (tr, va, te, mean, std, N) = data
    dl = lambda t, s: torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*t), batch_size=BATCH, shuffle=s)
    trL, vaL, teL = dl(tr, True), dl(va, False), dl(te, False)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lf = nn.HuberLoss(delta=HUBER_DELTA); best, bs, wait, t0, best_ep, ep_run = 1e18, None, 0, time.time(), 0, 0

    def predict(loader):
        ps, ts = [], []
        with torch.no_grad():
            for x, y, td, dw in loader:
                yp, _ = model(x.to(DEVICE), td.to(DEVICE), dw.to(DEVICE))
                ps.append((yp.cpu() * std + mean).numpy()); ts.append(y.numpy())
        return np.concatenate(ps), np.concatenate(ts)

    for ep in range(epochs):
        model.train(); ep_run = ep + 1
        for x, y, td, dw in trL:
            x, y, td, dw = x.to(DEVICE), y.to(DEVICE), td.to(DEVICE), dw.to(DEVICE)
            opt.zero_grad(); yp, aux = model(x, td, dw); loss = lf(yp * std + mean, y)
            if aux is not None:
                loss = loss + aux
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), CLIP); opt.step()
        sch.step(); model.eval()
        vm = metrics(*predict(vaL), horizons=[])['avg']['MAE']
        if vm < best:
            best, bs, wait, best_ep = vm, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0, ep + 1
        else:
            wait += 1
        log(f'[{name}] ep {ep + 1}/{epochs} valMAE={vm:.3f} ({time.time() - t0:.0f}s)')
        if wait >= PATIENCE:
            log(f'[{name}] early stop'); break
    model.load_state_dict(bs); model.eval()
    pred, true = predict(teL)
    res = dict(metrics=metrics(pred, true), params=int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
               minutes=(time.time() - t0) / 60, epochs_run=ep_run, best_epoch=best_ep, best_val_MAE=best,
               gpu=gpu_name(), torch=torch.__version__)
    if isinstance(model, DGTFNet):
        res['gate_g'] = model.gate_value()
        if model.adaptive and model.prior and model.gate_mode == 'time':
            res['gate_by_tod'] = time_gate_profile(model, teL)
            res['gate_g'] = res['gate_by_tod']['mean']         # mean of g_t over the test set
    if fz is not None:
        res['fuzzy'] = fuzzy_eval(pred, true, fz)
    return res, per_window_errors(pred, true)


def time_gate_profile(model, loader):
    """Mean learned g_t per hour of day (hour of the last input step) on the test set."""
    gs, hrs = [], []
    with torch.no_grad():
        for x, y, td, dw in loader:
            model(x.to(DEVICE), td.to(DEVICE), dw.to(DEVICE))
            gs.append(model._last_g.cpu().numpy()); hrs.append((td[:, -1].numpy() // 12))
    gs, hrs = np.concatenate(gs), np.concatenate(hrs)
    return {int(h): float(gs[hrs == h].mean()) for h in np.unique(hrs)} | {'mean': float(gs.mean()), 'std': float(gs.std())}


# ----------------------------------------------------------------------------------------
# 6. Job queue (one JSON + one NPZ per run; safe to resume; split across sessions)
# ----------------------------------------------------------------------------------------
def make_jobs(datasets, main_models, main_seeds, ablation_seeds, new_models, new_seeds, ablate=True):
    jobs = []
    for ds in datasets:                                        # priority 1: main comparison
        for m in main_models:
            for s in main_seeds:
                jobs.append(dict(ds=ds, kind='main', model=m, variant='', seed=s))
    if ablate:                                                 # priority 2: ablation (full model reused)
        for ds in datasets:
            for v in ABLATIONS:
                if v == 'Full DGTF-Net':
                    continue
                for s in ablation_seeds:
                    jobs.append(dict(ds=ds, kind='abl', model='DGTF-Net', variant=v, seed=s))
    for ds in datasets:                                        # priority 3: newer baselines
        for m in new_models:
            for s in new_seeds:
                jobs.append(dict(ds=ds, kind='main', model=m, variant='', seed=s))
    return jobs


def job_path(save_dir, j):
    tag = (j['variant'] or j['model']).replace('/', '').replace(' ', '_').replace('.', '')
    return os.path.join(save_dir, j['ds'], j['kind'], f"{tag}__seed{j['seed']}")


_CACHE = {}


def get_data(ds):
    if ds not in _CACHE:
        series, A, meta = load_dataset(ds)
        data = build_windows(series)
        fz = fuzzy_setup(series, int(series.shape[0] * TRAIN_FRAC))
        _CACHE[ds] = (data, torch.tensor(A).float(), fz, series)
    return _CACHE[ds]


def run_jobs(jobs, save_dir, worker=0, n_workers=1, log=print):
    todo = [j for i, j in enumerate(jobs) if i % n_workers == worker]
    for j in todo:
        out = job_path(save_dir, j)
        if os.path.exists(out + '.json'):
            continue
        os.makedirs(os.path.dirname(out), exist_ok=True)
        data, A, fz, _ = get_data(j['ds'])
        N = data[5]
        set_seed(j['seed'])
        if j['kind'] == 'abl':
            model = DGTFNet(N, A, **ABLATIONS[j['variant']])
        else:
            if j['model'] not in MODEL_REGISTRY:
                log(f"skip {j['model']}: not registered"); continue
            model = MODEL_REGISTRY[j['model']](N, A)
        name = f"{j['ds']}:{j['variant'] or j['model']}:s{j['seed']}"
        res, pw = train_eval(model, name, data, fz=fz, log=log)
        res.update(j); res['fuzzy_params'] = {k: v for k, v in fz.items() if k != 'cap'}
        np.savez_compressed(out + '.npz', pw=pw)
        json.dump(res, open(out + '.json', 'w'), indent=1)
        a = res['metrics']['avg']; log(f">>> {name}  avg MAE/RMSE/MAPE = {a['MAE']:.2f}/{a['RMSE']:.2f}/{a['MAPE']:.2f}%")
        del model; torch.cuda.empty_cache() if torch.cuda.is_available() else None


def persistence_results(save_dir, datasets):
    for ds in datasets:
        out = os.path.join(save_dir, ds, 'main', 'Persistence__seed0')
        if os.path.exists(out + '.json'):
            continue
        data, A, fz, _ = get_data(ds)
        (_, _, te, mean, std, N) = data
        xs, ys = te[0].numpy(), te[1].numpy()
        pred = np.repeat((xs[:, -1, :, 0] * std + mean)[:, None], Q, 1)
        res = dict(metrics=metrics(pred, ys), fuzzy=fuzzy_eval(pred, ys, fz), ds=ds, kind='main',
                   model='Persistence', variant='', seed=0, params=0,
                   fuzzy_params={k: v for k, v in fz.items() if k != 'cap'})
        os.makedirs(os.path.dirname(out), exist_ok=True)
        np.savez_compressed(out + '.npz', pw=per_window_errors(pred, ys)); json.dump(res, open(out + '.json', 'w'), indent=1)


# ----------------------------------------------------------------------------------------
# 7. Aggregation and statistics
# ----------------------------------------------------------------------------------------
def load_all(save_dir):
    rows = []
    for root, dirs, files in os.walk(save_dir):
        dirs[:] = [d for d in dirs if not d.startswith('_')]          # skip _summary etc.
        for f in files:
            if f.endswith('.json'):
                r = json.load(open(os.path.join(root, f)))
                if 'kind' not in r:
                    continue
                npz = os.path.join(root, f[:-5] + '.npz')
                r['_pw'] = np.load(npz)['pw'] if os.path.exists(npz) else None
                rows.append(r)
    return rows


def _key(r):
    return r['variant'] or r['model']


def dm_test(e1, e2, h=Q, power=1):
    """Diebold-Mariano test (two-sided) on loss series e1, e2 (per test window).
    HAC (Newey-West, Bartlett) variance with lag = P + Q - 1 because consecutive windows overlap.
    Returns (DM statistic, p-value). Negative DM -> model 1 has lower loss."""
    from scipy import stats
    d = np.abs(e1) ** power - np.abs(e2) ** power; T = len(d); dm = d.mean(); lag = P + Q - 1
    gamma = [np.sum((d[k:] - dm) * (d[:T - k] - dm)) / T for k in range(lag + 1)]
    var = gamma[0] + 2 * sum((1 - k / (lag + 1)) * gamma[k] for k in range(1, lag + 1))
    stat = dm / math.sqrt(var / T)
    return float(stat), float(2 * (1 - stats.norm.cdf(abs(stat))))


def holm(pvals):
    order = np.argsort(pvals); m = len(pvals); adj = np.empty(m); running = 0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i]); adj[i] = min(1.0, running)
    return adj.tolist()


def summarise(save_dir, out_dir, ref='DGTF-Net'):
    """Writes: main_table.csv, ablation_table.csv, significance.csv, fuzzy_table.csv, gate.csv, summary.json"""
    from scipy import stats
    os.makedirs(out_dir, exist_ok=True)
    rows = load_all(save_dir)
    summary = {}
    # ---- main comparison --------------------------------------------------------------
    main = [r for r in rows if r['kind'] == 'main']
    with open(os.path.join(out_dir, 'main_table.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'model', 'n_seeds', 'params'] + [f'{h}_{m}' for h in ['3', '6', '12', 'avg'] for m in ['MAE', 'RMSE', 'MAPE']])
        for ds in sorted({r['ds'] for r in main}):
            summary.setdefault(ds, {})
            for mdl in sorted({_key(r) for r in main if r['ds'] == ds}):
                rs = [r for r in main if r['ds'] == ds and _key(r) == mdl]
                cells = []
                for h in ['3', '6', '12', 'avg']:
                    for m in ['MAE', 'RMSE', 'MAPE']:
                        v = np.array([r['metrics'][h][m] for r in rs])
                        cells.append(f'{v.mean():.2f} ± {v.std(ddof=1) if len(v) > 1 else 0:.2f}')
                w.writerow([ds, mdl, len(rs), rs[0].get('params', '')] + cells)
                summary[ds][mdl] = {h: {m: [float(np.mean([r['metrics'][h][m] for r in rs])),
                                            float(np.std([r['metrics'][h][m] for r in rs], ddof=1)) if len(rs) > 1 else 0.0]
                                        for m in ['MAE', 'RMSE', 'MAPE']} for h in ['3', '6', '12', 'avg']}
                summary[ds][mdl]['n'] = len(rs); summary[ds][mdl]['params'] = rs[0].get('params')
    # ---- significance: DGTF-Net vs each other model ----------------------------------
    with open(os.path.join(out_dir, 'significance.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'baseline', 'DGTF mean MAE', 'baseline mean MAE', 'Welch t', 'Welch p', 'Welch p (Holm)',
                    'Mann-Whitney p', 'DM stat (seed-avg)', 'DM p', 'DM p (Holm)'])
        for ds in sorted({r['ds'] for r in main}):
            ref_rs = [r for r in main if r['ds'] == ds and _key(r) == ref]
            if not ref_rs:
                continue
            a = np.array([r['metrics']['avg']['MAE'] for r in ref_rs])
            ea = np.mean([r['_pw'].mean(1) for r in ref_rs], 0)
            lines = []
            for mdl in sorted({_key(r) for r in main if r['ds'] == ds} - {ref}):
                rs = [r for r in main if r['ds'] == ds and _key(r) == mdl]
                b = np.array([r['metrics']['avg']['MAE'] for r in rs])
                t, p = stats.ttest_ind(a, b, equal_var=False) if len(a) > 1 and len(b) > 1 else (np.nan, np.nan)
                u = stats.mannwhitneyu(a, b, alternative='two-sided').pvalue if len(a) > 1 and len(b) > 1 else np.nan
                eb = np.mean([r['_pw'].mean(1) for r in rs], 0)
                dms, dmp = dm_test(ea, eb)
                lines.append([ds, mdl, a.mean(), b.mean(), t, p, None, u, dms, dmp, None])
            for col, adjcol in ((5, 6), (9, 10)):
                ps = [l[col] for l in lines]
                ok = [i for i, v in enumerate(ps) if v == v]
                adj = holm([ps[i] for i in ok]) if ok else []
                for i, v in zip(ok, adj):
                    lines[i][adjcol] = v
            for l in lines:
                w.writerow([l[0], l[1]] + [f'{v:.4g}' if isinstance(v, (float, np.floating)) else v for v in l[2:]])
    # ---- ablation (paired by seed against the full model) ---------------------------
    abl = [r for r in rows if r['kind'] == 'abl']
    with open(os.path.join(out_dir, 'ablation_table.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'variant', 'n_seeds', 'avg MAE', '60-min MAE', 'Δ avg MAE vs full', 'paired t p', 'learned g'])
        for ds in sorted({r['ds'] for r in abl}):
            full = {r['seed']: r for r in main if r['ds'] == ds and _key(r) == ref}
            for v in ABLATIONS:
                rs = list(full.values()) if v == 'Full DGTF-Net' else [r for r in abl if r['ds'] == ds and r['variant'] == v]
                if not rs:
                    continue
                am = np.array([r['metrics']['avg']['MAE'] for r in rs]); m60 = np.array([r['metrics']['12']['MAE'] for r in rs])
                paired = [(r['metrics']['avg']['MAE'], full[r['seed']]['metrics']['avg']['MAE']) for r in rs if r['seed'] in full]
                if v != 'Full DGTF-Net' and len(paired) > 1:
                    x, y = zip(*paired); d = np.mean(np.array(x) - np.array(y)); p = stats.ttest_rel(x, y).pvalue
                else:
                    d, p = 0.0, np.nan
                g = [r.get('gate_g') for r in rs if r.get('gate_g') is not None]
                w.writerow([ds, v, len(rs), f'{am.mean():.2f} ± {am.std(ddof=1) if len(am) > 1 else 0:.2f}',
                            f'{m60.mean():.2f} ± {m60.std(ddof=1) if len(m60) > 1 else 0:.2f}', f'{d:+.2f}',
                            f'{p:.3g}' if p == p else '', f'{np.mean(g):.3f} ± {np.std(g):.3f}' if g else ''])
                summary.setdefault(ds, {}).setdefault('_ablation', {})[v] = dict(
                    avg=[float(am.mean()), float(am.std(ddof=1)) if len(am) > 1 else 0.0],
                    m60=[float(m60.mean()), float(m60.std(ddof=1)) if len(m60) > 1 else 0.0],
                    delta=float(d), p=None if p != p else float(p), n=len(rs),
                    g=[float(np.mean(g)), float(np.std(g))] if g else None,
                    gate_by_tod=[r.get('gate_by_tod') for r in rs if r.get('gate_by_tod')])
    # ---- fuzzy -------------------------------------------------------------------------
    with open(os.path.join(out_dir, 'fuzzy_table.csv'), 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['dataset', 'model', 'horizon', 'accuracy', 'macro-P', 'macro-R', 'macro-F1', 'F1 Low', 'F1 Med', 'F1 High', 'b1', 'b2'])
        for ds in sorted({r['ds'] for r in main}):
            for mdl in sorted({_key(r) for r in main if r['ds'] == ds}):
                rs = [r for r in main if r['ds'] == ds and _key(r) == mdl and 'fuzzy' in r]
                if not rs:
                    continue
                fp = rs[0].get('fuzzy_params') or {}
                for h in ['3', '6', '12', 'all']:
                    f = lambda k: np.mean([r['fuzzy'][h][k] for r in rs])
                    pc = np.mean([r['fuzzy'][h]['F1_per_class'] for r in rs], 0)
                    w.writerow([ds, mdl, h, f'{f("acc"):.4f}', f'{f("macro_P"):.4f}', f'{f("macro_R"):.4f}', f'{f("macro_F1"):.4f}',
                                f'{pc[0]:.4f}', f'{pc[1]:.4f}', f'{pc[2]:.4f}', f"{fp.get('b1', ''):.4}" if fp else '', f"{fp.get('b2', ''):.4}" if fp else ''])
                summary[ds][mdl]['fuzzy'] = {h: {k: float(np.mean([r['fuzzy'][h][k] for r in rs])) for k in ['acc', 'macro_F1']} for h in ['3', '6', '12', 'all']}
                summary[ds][mdl]['fuzzy_params'] = rs[0].get('fuzzy_params')
    # ---- run metadata -------------------------------------------------------------------
    meta = {r.get('gpu') for r in rows if r.get('gpu')}
    summary['_meta'] = dict(gpus=sorted(meta), torch=sorted({r.get('torch') for r in rows if r.get('torch')}),
                            minutes={f"{r['ds']}:{_key(r)}": r.get('minutes') for r in rows if r.get('seed') in (42, 0)},
                            epochs={f"{r['ds']}:{_key(r)}:s{r['seed']}": [r.get('epochs_run'), r.get('best_epoch')] for r in rows})
    json.dump(summary, open(os.path.join(out_dir, 'summary.json'), 'w'), indent=1, default=float)
    return summary


def make_figures(save_dir, out_dir, models_order=None):
    """fig_horizon_<ds>.png: MAE/RMSE/MAPE vs horizon (mean ± std over seeds);
       fig_ablation.png: avg MAE per ablation variant on each dataset (mean ± std)."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(out_dir, exist_ok=True)
    rows = load_all(save_dir); main = [r for r in rows if r['kind'] == 'main']
    for ds in sorted({r['ds'] for r in main}):
        mdls = models_order or sorted({_key(r) for r in main if r['ds'] == ds})
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
        for ax, met in zip(axes, ['MAE', 'RMSE', 'MAPE']):
            for mdl in mdls:
                rs = [r for r in main if r['ds'] == ds and _key(r) == mdl]
                if not rs:
                    continue
                v = np.array([[r['metrics'][str(h)][met] for h in range(1, Q + 1)] for r in rs])
                x = np.arange(1, Q + 1) * 5
                ax.errorbar(x, v.mean(0), yerr=v.std(0, ddof=1) if len(v) > 1 else None, label=mdl, capsize=2,
                            lw=2.2 if mdl == 'DGTF-Net' else 1.2, marker='o' if mdl == 'DGTF-Net' else None, ms=3)
            ax.set_xlabel('Horizon (min)'); ax.set_title(met + (' (%)' if met == 'MAPE' else '')); ax.grid(alpha=.3)
        axes[0].legend(fontsize=7, ncol=2); fig.suptitle(f'{ds}: test error by horizon (mean ± std over seeds)', fontsize=10)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, f'fig_horizon_{ds}.png'), dpi=300); plt.close(fig)
    abl = [r for r in rows if r['kind'] == 'abl']
    dss = sorted({r['ds'] for r in abl})
    if dss:
        fig, axes = plt.subplots(1, len(dss), figsize=(6.5 * len(dss), 3.8), squeeze=False)
        for ax, ds in zip(axes[0], dss):
            names, means, sds = [], [], []
            for v in ABLATIONS:
                rs = [r for r in main if r['ds'] == ds and _key(r) == 'DGTF-Net'] if v == 'Full DGTF-Net' else \
                     [r for r in abl if r['ds'] == ds and r['variant'] == v]
                if rs:
                    a = np.array([r['metrics']['avg']['MAE'] for r in rs])
                    names.append(v); means.append(a.mean()); sds.append(a.std(ddof=1) if len(a) > 1 else 0)
            y = np.arange(len(names))
            ax.barh(y, means, xerr=sds, color=['#C0392B' if n == 'Full DGTF-Net' else '#7F8C8D' for n in names], capsize=3)
            ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8); ax.invert_yaxis()
            lo = min(m - s for m, s in zip(means, sds)); hi = max(m + s for m, s in zip(means, sds))
            ax.set_xlim(lo - 0.15 * (hi - lo + 1e-6), hi + 0.15 * (hi - lo + 1e-6))
            ax.set_xlabel('Average test MAE (mean ± std over seeds)'); ax.set_title(ds); ax.grid(axis='x', alpha=.3)
        fig.tight_layout(); fig.savefig(os.path.join(out_dir, 'fig_ablation.png'), dpi=300); plt.close(fig)
