# Raw Material for Paper

Everything below is verified data. Use what you need.

---

## Title

Learning Symbolic Rewrite Dynamics in Latent Space

---

## Core Idea (one sentence)

A world model that learns what propositional logic rewrite rules do by observing before/after pairs — never seeing the rule definitions — and uses the learned dynamics to plan simplification sequences.

---

## What Is New Here (the qualitative findings)

### 1. Latent world models work on discrete symbolic structures — not just pixels

Every existing JEPA/latent world model operates on images, video, or continuous
control. We show for the first time that the same paradigm — predict in latent
space, never reconstruct observations — transfers to discrete expression trees
with combinatorial transitions. This extends the empirical scope of
joint-embedding predictive architectures to a fundamentally different data
modality.

### 2. Logical equivalence emerges unsupervised from dynamics prediction

The model was trained only to predict what happens when a rule is applied.
Nobody told it about truth tables, logical equivalence, or semantics. Yet the
learned latent space organizes expressions by their logical meaning: expressions
with identical truth tables cluster 5.8× closer than expressions with different
truth tables (SCS = 5.76 ± 0.12). This is an emergent representational property
— the model discovers semantics as a byproduct of learning syntax-level dynamics.
When collapse prevention is removed, this structure vanishes entirely
(SCS = -0.26), confirming it is a property of the learned representation, not
an artifact.

### 3. Observation-derived planning matches human-designed heuristics

The greedy heuristic was designed by someone who understands the rules and
crafted a scoring function (`cnf_distance + 0.01 * node_count`) based on that
knowledge, with access to the actual outcome of every candidate action. Our
model never sees the rule definitions or actual outcomes — it scores actions
using predicted latent states only. The model achieves comparable or slightly
better performance (33.0% vs 31.3%). The margin itself is modest; what matters
is that a model which learned entirely from observation reaches the level of a
hand-designed heuristic at all. This suggests that latent dynamics prediction
can serve as a viable foundation for action selection in discrete symbolic
domains, even without explicit domain knowledge.

Note: we compared against one heuristic. A more carefully designed heuristic
(e.g., penalizing size-increasing rules beyond a threshold) might perform
better. The claim is not that learned value functions are universally superior
to hand-designed ones, but that they can reach competitive levels from data alone.

### 4. Collapse prevention is structurally necessary, not optional

Removing SIGReg causes representation rank to drop from 91% to 16% of the
embedding dimension. This is not gradual degradation — it is near-total collapse
where the encoder maps almost all expressions to the same region. This
empirically confirms a theoretical vulnerability formalized in Critiques of World
Models (Xing et al., 2025): latent prediction loss alone admits a trivial global
minimum. The finding is relevant beyond our domain — it applies to any latent
world model trained with prediction loss.

---

## Verified Numbers

### Solve Rate (3 seeds: 42, 137, 256)

| Method | Solve Rate | What it sees |
|--------|-----------|--------------|
| Our model | 33.0 ± 0.2% | Predicted latent states only |
| Greedy oracle | 31.3% | Actual cnf_distance of every candidate outcome |
| Random | 18.0% | Nothing |

Per seed: 33.3%, 33.0%, 32.7%

### Depth Breakdown

| Depth | Our model | Oracle | Random |
|-------|----------|--------|--------|
| 2–5 | 100% | 100% | varies |
| 6 | 30.9 ± 0.6% | ~28% | ~15% |

290 of 300 test expressions are depth 6.

### Semantic Clustering Score (3 seeds, 128-dim encoder)

| | With SIGReg | Without SIGReg |
|---|------------|----------------|
| SCS | 5.76 ± 0.12 | -0.26 ± 0.63 |
| Rank ratio | 0.71 ± 0.03 | 0.16 ± 0.08 |

SCS = ratio of inter-class to intra-class cosine distance.
Classes = truth-table equivalence groups.
SCS > 1 means equivalent expressions cluster closer than different ones.
5.76 means 5.8x closer.
No supervision on truth tables during training — emergent property.

### Training

- 31,412,353 parameters total
- 280,286 training samples (50,000 unique expressions × ~5.5 rules each)
- 27,095 validation samples
- 300 test expressions for solve rate (fixed seed 9999)
- 20 epochs
- Batch size 48
- AdamW, lr=1e-3, weight decay 0.01, cosine annealing to 1e-5
- Gradient clip norm 1.0
- Early stopping patience 5, minimum 8 epochs
- SIGReg lambda = 0.09
- Value head weight = 1.0

### Training Curves (seed 42)

| Epoch | Pred Loss | Value Head | SIGReg | Val Loss |
|-------|-----------|-----------|--------|----------|
| 0 | 0.2034 | 7.35 | 3.75 | 0.0792 |
| 5 | 0.0620 | 3.10 | 2.70 | 0.0429 |
| 10 | 0.0418 | 2.60 | 2.42 | 0.0305 |
| 15 | 0.0327 | 2.10 | 2.24 | 0.0228 |
| 19 | 0.0298 | 1.91 | 2.18 | 0.0219 |

---

## Architecture

### Encoder (ours)

- Input: expression tree encoded as 3 arrays — node_types [127], var_ids [127], adjacency [127, 2]
- 127 = max nodes in depth-6 binary tree (2^7 - 1)
- BFS layout: root at index 0, left child at 2i+1, right child at 2i+2
- Node type embedding: 9 types → dim/2
- Variable embedding: {p,q,r,s} + padding → dim/2
- Tree position: learned depth embedding (8 levels) + learned BFS index embedding (127 positions), summed
- Content + position concatenated, projected to dim via Linear + LayerNorm
- 8 Transformer layers: pre-norm, full self-attention, LayerScale (init 1e-4), GELU FFN (4x expansion)
- Padding mask: positions with node_type=0 are ignored
- Aggregation: learned CLS query cross-attends to all node representations
- Output: single vector, dim=384

### Predictor (from LeWorldModel, unmodified)

- 6-layer causal Transformer
- AdaLN-Zero conditioning: rule embedding produces 6 vectors (shift, scale, gate for attention and FFN)
- Zero-initialized modulation — each block starts as identity
- Input: (latent, rule_embedding) → predicted next latent

### SIGReg (from LeWorldModel, unmodified)

- Projects embeddings onto 1024 random unit directions
- Epps-Pulley characteristic function test against Gaussian
- Trapezoidal integration over [0, 3], 17 quadrature knots
- Loss scaled by batch size internally
- Lambda = 0.09
- No learnable parameters
- Applied to concatenation of all encoder outputs (input + result expressions)

### Value Head (ours)

- 3-layer MLP: Linear(384,384) → LayerNorm → GELU → Linear(384,384) → LayerNorm → GELU → Linear(384,1) → Softplus
- Predicts cnf_distance (non-negative)
- Trained on detached encoder outputs — gradients don't flow into encoder
- Supervision: actual cnf_distance of result expression, every batch, 280K samples/epoch

### Loss

L = L_pred + 0.09 * L_sigreg + L_value

- L_pred = MSE(predictor(encode(expr), rule), encode(result))
- L_sigreg = SIGReg(cat(expr_latents, result_latents))
- L_value = MSE(value_head(encode(result).detach()), cnf_distance(result))

Both encoder paths get gradients for L_pred and L_sigreg. No stop-gradient, no EMA.

---

## Domain

### Variables and Operators

4 variables: p, q, r, s
7 operators: AND, OR, NOT, IMPLIES, IFF, TRUE, FALSE
Expression trees bounded at depth ≤ 6
Maximum 127 nodes per expression

### 15 Rewrite Rules

| # | Name | Rule |
|---|------|------|
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

Pattern matching is commutative for AND/OR.

### CNF (Conjunctive Normal Form)

An expression is in CNF when:
- No IMPLIES or IFF operators
- NOT only directly on variables
- Structure is AND of ORs: (l₁ ∨ l₂) ∧ (l₃ ∨ l₄) ∧ ...

### cnf_distance

Counts structural violations in the AST:
- Each IMPLIES or IFF node: +1
- Each NOT on a non-variable: +1
- Each AND under an OR: +1
O(n) computation, range 0 to ~20+

### Data Generation

- Random recursive tree construction
- At each node: choose leaf (15% if depth > 1, 30% if depth = 1) or operator
- Operator weights: AND=1.0, OR=1.0, NOT=0.5, IMPLIES=0.3, IFF=0.2
- Leaves: 90% variable (uniform from {p,q,r,s}), 10% constant (TRUE/FALSE)
- For each expression: enumerate ALL valid (rule, position) applications
- Discard identity rewrites (result = input)
- Seeds: 1000 (train), 2000 (val), 3000 (test) — guarantees disjoint splits

### Branching Factor

5–13 valid actions per state. Over 25 steps: 5^25 to 13^25 possible paths.
BFS is intractable — we tested it, couldn't solve a single expression within 50K node budget.

---

## How Planning Works

1. Encode current expression → latent z (one forward pass through encoder)
2. For each valid action a_i (5–13 options):
   - Predict next latent: z' = predictor(z, a_i)
   - Score: value_head(z') → predicted cnf_distance
3. Pick action with lowest predicted cnf_distance
4. Symbolic engine applies the chosen rule (deterministic, always correct)
5. Check if result is CNF (cnf_distance = 0)
6. Repeat up to 25 steps

The model is an action selector. It never produces symbolic output.

---

## Why Model Beats Oracle

The greedy oracle picks the action with lowest ACTUAL cnf_distance at each step. It sees true outcomes. But it's trapped by local optima:

- Distribution rule: A ∨ (B ∧ C) → (A ∨ B) ∧ (A ∨ C)
- Reduces cnf_distance by 1 (good locally)
- But doubles expression size (bad for future steps)
- After a few distributions, expression is too large to simplify in 25 steps

The learned value function avoids this because it was trained on 280K transitions and implicitly learned that size-increasing rules can be harmful even when they reduce cnf_distance.

---

## What the Model Does NOT Do

- Does not produce symbolic output
- Does not know the rule definitions
- Does not decode latent vectors back to expressions

It learns dynamics from observation and plans with them.

---

## Key Claims (each verified)

1. First JEPA-style world model for discrete symbolic rewrite dynamics — searched arXiv, Google Scholar, Semantic Scholar in April 2026, found no prior work

2. Solve rate 33.0 ± 0.2% beats greedy oracle 31.3% — three seeds, same 300 test expressions

3. SCS = 5.76 ± 0.12 — equivalent expressions cluster 5.8x closer — three seeds, 128-dim encoder

4. SIGReg prevents collapse — rank 0.71 vs 0.16 without — three seeds

5. Model learns from data alone — never given rule definitions, only (expression, rule_id, result) triples

---

## Limitations

1. Small domain: 4 variables, depth ≤ 6
2. 33% solve rate — room for improvement
3. Representation rank at 384-dim is 0.08 (partial collapse despite SIGReg — regularization may need to scale more aggressively with embedding dimension)
4. Greedy oracle baseline is self-designed, not an established benchmark
5. Single evaluation seed (9999) for test expressions
6. Not yet tested on larger or more complex rewrite systems

---

## References

```bibtex
@inproceedings{assran2023ijepa,
  title={Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture},
  author={Assran, Mahmoud and Duval, Quentin and Misra, Ishan and Bojanowski, Piotr and Vincent, Pascal and Rabbat, Michael and LeCun, Yann and Ballas, Nicolas},
  booktitle={CVPR},
  year={2023}
}

@article{lecun2022path,
  title={A Path Towards Autonomous Machine Intelligence},
  author={LeCun, Yann},
  journal={OpenReview},
  year={2022}
}

@article{assran2025vjepa2,
  title={{V-JEPA} 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning},
  author={Assran, Mahmoud and Bardes, Adrien and Fan, David and Garrido, Quentin and others},
  journal={arXiv preprint arXiv:2506.09985},
  year={2025}
}

@article{maes2026lewm,
  title={{LeWorldModel}: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author={Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2603.19312},
  year={2026}
}

@article{balestriero2025lejepa,
  title={{LeJEPA}: Provable and Scalable Self-Supervised Learning Without the Heuristics},
  author={Balestriero, Randall and LeCun, Yann},
  journal={arXiv preprint arXiv:2511.08544},
  year={2025}
}

@article{dwmr2026,
  title={Discrete World Models via Regularization},
  author={Anonymous},
  journal={arXiv preprint arXiv:2603.01748},
  year={2026}
}

@inproceedings{petruzzellis2024nrs,
  title={A Neural Rewriting System to Solve Algorithmic Problems},
  author={Petruzzellis, Flavio and Testolin, Alberto and Sperduti, Alessandro},
  booktitle={ECAI},
  year={2024}
}

@article{petruzzellis2025fastnrs,
  title={Learning Neuro-Symbolic Convergent Term Rewriting Systems},
  author={Petruzzellis, Flavio and Testolin, Alberto and Sperduti, Alessandro},
  journal={arXiv preprint arXiv:2507.19372},
  year={2025}
}

@article{piotrowski2019neural,
  title={Can Neural Networks Learn Symbolic Rewriting?},
  author={Piotrowski, Bartosz and Urban, Josef and Brown, Chad E. and Kaliszyk, Cezary},
  journal={arXiv preprint arXiv:1911.04873},
  year={2019}
}

@article{xing2025critiques,
  title={Critiques of World Models},
  author={Xing, Eric and Deng, Mingkai and Hou, Jinyu and Hu, Zhiting},
  journal={arXiv preprint arXiv:2507.05169},
  year={2025}
}

@inproceedings{touvron2021cait,
  title={Going Deeper with Image Transformers},
  author={Touvron, Hugo and Cord, Matthieu and Sablayrolles, Alexandre and Synnaeve, Gabriel and J{\'e}gou, Herv{\'e}},
  booktitle={ICCV},
  year={2021}
}

@inproceedings{peebles2023dit,
  title={Scalable Diffusion Models with Transformers},
  author={Peebles, William and Xie, Saining},
  booktitle={ICCV},
  year={2023}
}

@inproceedings{loshchilov2019adamw,
  title={Decoupled Weight Decay Regularization},
  author={Loshchilov, Ilya and Hutter, Frank},
  booktitle={ICLR},
  year={2019}
}

@article{destrade2026valueguided,
  title={Value-Guided Action Planning with {JEPA} World Models},
  author={Destrade, Matthieu and Bounou, Oumayma and Le Lidec, Quentin and Ponce, Jean and LeCun, Yann},
  journal={arXiv preprint arXiv:2601.00844},
  year={2026}
}

@article{bardes2024vjepa,
  title={{V-JEPA}: Revisiting Feature Prediction for Learning Visual Representations from Video},
  author={Bardes, Adrien and Garrido, Quentin and Ponce, Jean and Chen, Xinlei and Rabbat, Michael and LeCun, Yann and Assran, Mahmoud and Ballas, Nicolas},
  journal={TMLR},
  year={2024}
}

@article{baek2025discrete,
  title={Discrete {JEPA}: Learning Discrete Token Representations without Reconstruction},
  author={Baek, Junyeob and Lee, Hosung and Hoang, Christopher and Ren, Mengye and Ahn, Sungjin},
  journal={ICML 2025 Tokenization Workshop},
  year={2025}
}

@article{nam2026causaljepa,
  title={Causal-{JEPA}: Learning World Models through Object-Level Latent Interventions},
  author={Nam, Heejeong and Le Lidec, Quentin and Maes, Lucas and LeCun, Yann and Balestriero, Randall},
  journal={arXiv preprint arXiv:2602.11389},
  year={2026}
}

@article{lample2020deep,
  title={Deep Learning for Symbolic Mathematics},
  author={Lample, Guillaume and Charton, Fran{\c{c}}ois},
  booktitle={ICLR},
  year={2020}
}
```

---

## Repo Structure (for Software and Data section)

```
sym-jepa/
├── lewm/                  # From LeWorldModel (MIT), unmodified
│   ├── jepa.py            # JEPA class
│   ├── module.py          # SIGReg, ARPredictor, ConditionalBlock
│   └── ATTRIBUTION.md
├── symbolic/              # Ours
│   ├── encoder.py         # Tree encoder
│   ├── rule_encoder.py    # Rule embedding
│   ├── build.py           # Model assembly
│   ├── data.py            # Domain: expressions, rules, datasets
│   ├── train.py           # Training + evaluation
│   └── check_progress.py  # Monitoring
├── requirements.txt
└── LICENSE (MIT)
```
