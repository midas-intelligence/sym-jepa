# Sym-JEPA: Latent World Model for Symbolic Rewrite Dynamics

A latent world model that learns the dynamics of propositional logic rewrite rules purely from data — the model is never given the rules explicitly. It observes (expression, rule_id, result) triples, learns to predict expression transformations in latent space, and uses a learned value function to plan simplification sequences toward Conjunctive Normal Form (CNF).

## Motivation

Can a neural world model learn rewrite dynamics from observation alone, and use those learned dynamics to plan?

The model receives expression trees as structured input and a rule ID as an integer. It never sees the rule definitions. It learns what each rule does by observing before/after pairs, then uses that knowledge to select actions during planning.

## Results

Evaluated on 300 randomly generated non-CNF expressions (depth 2–6, 4 variables). Each method selects which rewrite rule to apply at each step; a symbolic engine executes the rewrite. Success = reaching CNF within 25 steps. Results reported as mean ± std across 3 training seeds (42, 137, 256).

| Method | Solve Rate | How it picks actions |
|--------|-----------|----------------------|
| **Sym-JEPA** | **33.0 ± 0.2%** | Learned value function scores predicted latent states |
| Greedy heuristic | 31.3% | Computes actual cnf_distance of every candidate outcome |
| Random | 18.0% | Uniform random selection |

31.4M parameters. Trained for 20 epochs.

The greedy heuristic has access to ground-truth outcomes that our model never sees — it computes the real cnf_distance after every candidate action. Our model scores actions using only predicted latent states. Despite this information disadvantage, the learned value function achieves higher solve rate because it develops implicit lookahead: it avoids actions that look good immediately but create dead ends downstream (e.g., distribution rules that reduce cnf_distance by 1 but double expression size, exhausting the step budget).

### Representation Properties (128-dim encoder, 3 seeds)

| Metric | With SIGReg | Without SIGReg |
|--------|------------|----------------|
| Semantic Clustering Score | 5.76 ± 0.12 | -0.26 ± 0.63 |
| Representation rank ratio | 0.71 ± 0.03 | 0.16 ± 0.08 |

The model discovers logical equivalence as an emergent property: expressions with identical truth tables cluster 5.8× closer than logically distinct ones, without any supervision on truth tables. This structure vanishes without SIGReg, confirming that distributional regularization is necessary to prevent representation collapse.

## Architecture

```
Expression tree ──► TreeEncoder (8-layer Transformer) ──► latent
                                                            │
Rule ID ──► RuleEncoder ──► AdaLN conditioning             │
                                │                           │
                    ARPredictor (6-layer causal Transformer) │
                                │                           │
                         predicted next latent              │
                                │                           │
                         ValueHead ──► cnf_distance estimate
```

- **TreeEncoder**: Transformer over BFS-encoded expression trees with tree-aware positional encoding (depth + BFS index). Full self-attention across all 127 positions. LayerScale initialization. Learned CLS query aggregation.
- **ARPredictor**: Causal Transformer with AdaLN-Zero conditioning — rule embedding modulates attention and FFN via adaptive LayerNorm. From [LeWorldModel](https://github.com/lucas-maes/le-wm).
- **SIGReg**: Distributional regularizer enforcing isotropic Gaussian embeddings via random projections and Epps-Pulley characteristic function test. From [LeWorldModel](https://github.com/lucas-maes/le-wm).
- **ValueHead**: MLP predicting cnf_distance from latent vectors. Trained on detached encoder outputs.

### Loss

```
L = L_pred + λ · L_sigreg + L_value
```

Both encoder paths receive gradients (no stop-gradient, no EMA). Value head trains on detached encoder outputs.

## Domain

Propositional logic over {p, q, r, s}. 15 rewrite rules toward CNF: implication/biconditional elimination, De Morgan's laws, distribution, double negation, complement, identity, annihilation, idempotency. Expression trees bounded at depth ≤ 6.

## Repository Structure

```
├── lewm/                  # From LeWorldModel (MIT license, unmodified)
│   ├── jepa.py            # JEPA class
│   ├── module.py          # SIGReg, ARPredictor, ConditionalBlock
│   └── ATTRIBUTION.md
├── symbolic/              # Domain-specific (ours)
│   ├── encoder.py         # TreeEncoder
│   ├── rule_encoder.py    # Rule embedding
│   ├── build.py           # Model assembly
│   ├── data.py            # Expressions, rules, dataset generation
│   ├── train.py           # Training + solve rate evaluation
│   └── check_progress.py  # Monitoring
├── requirements.txt
└── LICENSE
```

## Usage

```bash
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

# Evaluate solve rate
python -m symbolic.check_progress --size large --seed 42 --solve-rate
```

## Acknowledgments

Uses SIGReg and ARPredictor from [LeWorldModel](https://github.com/lucas-maes/le-wm) by Maes, Le Lidec, Scieur, LeCun & Balestriero (arXiv 2603.19312, 2026), licensed under MIT.

## License

MIT
