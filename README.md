# Sym-JEPA: Latent World Model for Symbolic Rewrite Dynamics

A latent world model that learns the dynamics of propositional logic rewrite rules purely from data — the model is never given the rules explicitly. It observes (expression, rule_id, result) triples, learns to predict expression transformations in latent space, and uses a learned value function to plan simplification sequences toward Conjunctive Normal Form (CNF).

Uses SIGReg and ARPredictor components from [LeWorldModel](https://github.com/lucas-maes/le-wm) (Maes et al., 2026).

## Motivation

Symbolic simplification is typically solved by hardcoded algorithms or LLMs that already know the rules. We ask a different question: **can a neural world model learn rewrite dynamics from observation alone, and use those learned dynamics to plan?**

The model receives expression trees as structured input and a rule ID as an integer. It never sees the rule definitions. It learns what each rule does by observing before/after pairs, then uses that knowledge to select actions during planning.

## Results

Evaluated on 300 randomly generated non-CNF expressions (seed 9999, depth 2–6, 4 variables). Each method selects which rewrite rule to apply at each step; a symbolic engine executes the rewrite. Success = reaching CNF within 25 steps.

| Method | Solve Rate | How it picks actions |
|--------|-----------|----------------------|
| **Sym-JEPA** | **34.0%** | Learned value function scores predicted latent states |
| Greedy oracle | 31.3% | Sees the actual outcome of every action, picks lowest cnf_distance |
| Random | 18.0% | Uniform random selection |

31.4M parameters. Trained for 20 epochs.

**Sym-JEPA outperforms the greedy oracle despite having strictly less information.** The greedy oracle cheats — it computes the real cnf_distance of every candidate result before choosing. Our model never sees the actual outcome; it scores actions using predicted latent states from the world model. The greedy oracle is trapped by local optima (e.g., the distribution rule reduces cnf_distance by 1 but adds 20 nodes, creating a dead end). The learned value function avoids these traps because it was trained on 280K expression transitions and develops implicit lookahead.

## Architecture

### Overview

```
INPUT:
  Expression tree ──► TreeEncoder ──► expr_latent ──┐
  Rule ID ──► RuleEncoder ──► AdaLN conditioning    │
                                  │                  │
                      ARPredictor (causal Transformer)
                                  │
                           pred_latent
                                  │
                           ValueHead ──► estimated cnf_distance

TARGET:
  Result tree ──► TreeEncoder ──► target_latent (same encoder, both get gradients)

LOSS:
  L2(pred_latent, target_latent) + λ·SIGReg(all_latents) + value_MSE
```

### TreeEncoder (custom — `symbolic/encoder.py`)

Transforms BFS-encoded expression trees into latent vectors via Transformer self-attention.

- **Input**: Three arrays per expression — `node_types [127]` (what each node is: AND, OR, NOT, VAR, etc.), `var_ids [127]` (which variable: p/q/r/s), `adjacency [127, 2]` (tree structure)
- **Position encoding**: Learned depth embedding (0–7) + learned BFS index embedding, summed
- **Content embedding**: Node type embedding (dim/2) + variable embedding (dim/2), concatenated
- **Layers**: 8 Transformer layers with pre-norm, LayerScale (init 1e-4), GELU FFN
- **Aggregation**: Learned CLS query cross-attends to all node representations
- **Output**: Single latent vector (384-dim for Large model)

Every node attends to every other node — the attention learns which structural relationships matter, rather than being constrained to parent-child edges.

### ARPredictor (from LeWorldModel — `lewm/module.py`)

Causal Transformer that predicts the next latent state given current state and action.

- **Conditioning**: AdaLN-zero modulation — the rule embedding modulates attention and FFN via adaptive LayerNorm, rather than being concatenated to the input. Zero-initialized so each block starts as identity.
- **Layers**: 6 ConditionalBlock layers, each with causal self-attention + conditioned FFN
- **Input**: (latent, rule_embedding) → predicted next latent

### SIGReg (from LeWorldModel — `lewm/module.py`)

Sketched Isotropic Gaussian Regularization prevents representation collapse.

- Projects embeddings along 1024 random unit directions
- Computes Epps-Pulley characteristic function test against a Gaussian reference
- Uses trapezoidal integration with Gaussian window over [0, 3]
- Loss scaled by batch size internally
- Lambda = 0.09 (LeWorldModel's default)

### ValueHead (custom — `symbolic/train.py`)

MLP that predicts cnf_distance from a latent vector, providing a planning compass.

- 3-layer MLP: Linear → LayerNorm → GELU → Linear → LayerNorm → GELU → Linear → Softplus
- Trained on every batch using actual cnf_distance of result expressions (280K samples/epoch)
- During planning: scores each candidate action by predicted cnf_distance of its predicted outcome

### Model Configurations

| Size | Encoder dim | Enc layers | Heads | Pred layers | Total params |
|------|-------------|-----------|-------|-------------|-------------|
| Large | 384 | 8 | 8 | 6 | 31.1M |
| XL | 512 | 8 | 8 | 6 | 33.5M |

## Domain

### Propositional Logic

Expressions over variables {p, q, r, s} with operators {AND, OR, NOT, IMPLIES, IFF, TRUE, FALSE}. Expression trees bounded at depth ≤ 6 (max 127 nodes).

### 15 Rewrite Rules

| ID | Name | Pattern → Result |
|----|------|------------------|
| 0 | impl_elim | A → B ↦ ¬A ∨ B |
| 1 | bicond_elim | A ↔ B ↦ (¬A ∨ B) ∧ (¬B ∨ A) |
| 2 | double_neg | ¬¬A ↦ A |
| 3 | demorgan_and | ¬(A ∧ B) ↦ ¬A ∨ ¬B |
| 4 | demorgan_or | ¬(A ∨ B) ↦ ¬A ∧ ¬B |
| 5 | neg_true | ¬⊤ ↦ ⊥ |
| 6 | neg_false | ¬⊥ ↦ ⊤ |
| 7 | dist_or_and | A ∨ (B ∧ C) ↦ (A ∨ B) ∧ (A ∨ C) |
| 8 | complement_and | A ∧ ¬A ↦ ⊥ |
| 9 | complement_or | A ∨ ¬A ↦ ⊤ |
| 10 | identity_and | A ∧ ⊤ ↦ A |
| 11 | identity_or | A ∨ ⊥ ↦ A |
| 12 | annihilate_and | A ∧ ⊥ ↦ ⊥ |
| 13 | annihilate_or | A ∨ ⊤ ↦ ⊤ |
| 14 | idempotent | A ∧ A ↦ A, A ∨ A ↦ A |

Pattern matching is commutative for AND/OR (e.g., complement_and matches both A ∧ ¬A and ¬A ∧ A).

### Canonical Form (CNF)

An expression is in simplified CNF when:
- No IMPLIES or IFF operators remain
- NOT appears only directly on variables (negation normal form)
- Structure is AND of ORs: (l₁ ∨ l₂) ∧ (l₃ ∨ l₄) ∧ ...

### Reward Signal

`cnf_distance(expr)` counts structural violations of CNF form in the AST. O(n) computation, no external tools. Range: 0 (is CNF) to ~20+ (deeply nested non-CNF).

### Dataset

All data synthetically generated. Deterministic with fixed seeds.

| Split | Unique Expressions | Total Samples | Seed |
|-------|-------------------|---------------|------|
| Train | 50,000 | 280,286 | 1000 |
| Val | 5,000 | 27,095 | 2000 |
| Test | 10,000 | 53,766 | 3000 |

Each sample is an (expression, rule_id, position, result) tuple. ~5.5 valid rule applications per expression on average. No expression appears in multiple splits.

## How Planning Works

The model is an **action selector**, not a symbolic solver. At each step:

1. Encode current expression → latent vector
2. For each valid (rule, position) action:
   - Predict next latent via the predictor
   - Score it with the value head (predicted cnf_distance)
3. Pick the action with the lowest predicted cnf_distance
4. Apply that action symbolically (Python rule engine does the actual rewrite)
5. Check if result is CNF
6. Repeat for up to 25 steps

The model decides **which rule to apply**. The symbolic engine executes the rewrite. Success = reaching CNF within 25 steps.

## Training

### Loss Function

```
L = L_pred + λ · L_sigreg + L_value
```

| Component | Formula | Purpose |
|-----------|---------|---------|
| L_pred | mean((pred_latent - target_latent)²) | Learn rewrite dynamics |
| L_sigreg | SIGReg(cat(expr_latents, target_latents)) | Prevent collapse |
| L_value | mean((value_head(target_latent) - cnf_distance)²) | Learn planning signal |

Both encoder paths receive gradients (no stop-gradient, no EMA — per LeJEPA design). Value head trains on detached encoder outputs to avoid interfering with prediction.

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=0.01) |
| Scheduler | Cosine annealing (min_lr=1e-5) |
| Batch size | 48 |
| Max epochs | 20 |
| Early stopping | patience=5, min_epochs=8 |
| Gradient clip | max_norm=1.0 |
| SIGReg lambda | 0.09 |
| Value weight | 1.0 |

## Repository Structure

```
sym-jepa/
├── lewm/                      # From LeWorldModel (MIT license, unmodified)
│   ├── __init__.py
│   ├── jepa.py                # JEPA class: encode, predict, rollout, criterion, get_cost
│   └── module.py              # SIGReg, ARPredictor, ConditionalBlock, Attention,
│                              #   FeedForward, Transformer, Embedder, MLP
├── symbolic/                  # Domain-specific (ours)
│   ├── __init__.py
│   ├── encoder.py             # TreeEncoder: Transformer over BFS-encoded expression trees
│   ├── rule_encoder.py        # nn.Embedding for discrete rewrite rule IDs
│   ├── build.py               # Assembles JEPA with tree/rule adapters
│   ├── data.py                # Expr class, 15 rewrite rules, cnf_distance, dataset generation,
│   │                          #   tensor encoding, truth table evaluation
│   ├── train.py               # Full training loop + solve rate evaluation
│   └── check_progress.py      # Load checkpoint, display curves, evaluate solve rate
├── data/                      # Generated datasets (not tracked)
├── checkpoints/               # Model weights (not tracked)
├── logs/                      # Training logs (not tracked)
├── requirements.txt
├── LICENSE                    # MIT
└── README.md
```

## Usage

```bash
# Install
pip install -r requirements.txt

# Generate data
python -c "
from symbolic.data import generate_dataset
import pickle, os
os.makedirs('data', exist_ok=True)
for name, n, seed in [('train', 50000, 1000), ('val', 5000, 2000), ('test', 10000, 3000)]:
    samples = generate_dataset(num_expressions=n, seed=seed)
    with open(f'data/{name}.pkl', 'wb') as f: pickle.dump(samples, f)
    print(f'{name}: {len(samples)} samples')
"

# Train
python -u -m symbolic.train --size large --seed 42 --max-epochs 20

# Check training progress
python -m symbolic.check_progress --size large --seed 42

# Evaluate solve rate on latest checkpoint
python -m symbolic.check_progress --size large --seed 42 --solve-rate
```

## Acknowledgments

This project uses SIGReg and ARPredictor from [LeWorldModel](https://github.com/lucas-maes/le-wm) by Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun & Randall Balestriero (arXiv 2603.19312, 2026), licensed under MIT.

Architecture informed by:
- [LeJEPA](https://arxiv.org/abs/2511.08544) (Balestriero & LeCun, 2025) — SIGReg collapse prevention
- [I-JEPA](https://arxiv.org/abs/2301.08243) (Assran et al., CVPR 2023) — Joint-embedding predictive architecture
- [V-JEPA 2](https://arxiv.org/abs/2506.09985) (Assran et al., 2025) — Latent planning via MPC

## License

MIT
