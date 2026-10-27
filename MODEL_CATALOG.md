# QMASE Model Catalog and Maturity

This catalog separates implemented, limited-scope analysis from research
proposals. A model being listed here does not mean it is trained, validated,
deployed, connected to production, or integrated with BIOGAS. The current
engine is a local research scaffold; it does not produce a validated universal
martial-arts score or autonomous coaching decisions.

## Implemented or available as exploratory components

| Model/component | Current implementation | Limits |
|---|---|---|
| Bayesian binomial mixed-effects logistic model | Optional fit with an athlete-level random intercept and discipline/ruleset fixed effects in `src/martial_arts_engine/mixed_models.py`. | Requires repeated labeled binary outcomes and multiple athletes/disciplines. Not a personal ability score, causal effect, cross-classified model, or general multilevel model. |
| Cox proportional-hazards model | Optional injury follow-up analysis in `src/martial_arts_engine/survival.py`. | Requires a valid cohort, exposure definition, censoring, confounder review, and enough events. Exploratory association only; no individual medical prediction. |
| Markov transition summaries | Transition counts and row-normalized probabilities in `src/martial_arts_engine/transitions.py`; a synthetic demo is provided. | Descriptive first-order transition summaries, not a fitted semi-Markov process, tactical policy, or evidence of real-world performance. |
| Observed dwell summaries | Per-state observed duration summaries from event timestamps in `transitions.py`. | Sequence-end censoring is recorded, but the summaries are not a survival-adjusted dwell model or causal estimate. |
| Pose and kinematic features | MediaPipe pose extraction and projected image-space geometry/speed features in `vision.py` and `kinematics.py`. | Camera-plane features are not force, validated technique labels, or injury-risk measures. |
| Pose-feature baseline | Optional frame-feature MLP in `deep_learning.py`; a separate exploratory ViCoS pose-position classifier/training workflow exists. | Not a temporal transformer, universal technique detector, or validated production model. ViCoS use is non-commercial and limited to the approved BJJ position task. |
| Wearable telemetry summaries | Consent- and provenance-checked descriptive summaries in `wearable_telemetry.py`. | No concussion threshold, cumulative brain injury, fatigue diagnosis, or clinical interpretation. |
| Hawkes point process (exponential kernel) | Maximum-likelihood fit of position-change timing vs. a Poisson baseline, with shared and per-session baselines, in `hawkes.py` (`fit-vicos-hawkes`). | Exploratory, ViCoS only (two athletes, six sessions). With per-session baselines the branching ratio is ≈ 0, so no self-excitation is found; it cannot separate self-excitation from match-phase rate changes, and says nothing about one athlete's action causing an opponent's mistake. |

## Proposed research roadmap (not implemented)

| Model/system | Potential QMASE use | Prerequisites and cautions |
|---|---|---|
| Linear mixed-effects models (LMM) | Continuous repeated outcomes with athlete/session clustering. | Define a valid outcome, repeated-measures design, missingness strategy, and diagnostics. |
| Hierarchical Bayesian multilevel models | Partial pooling across athletes, coaches, gyms, or organizations. | Pre-register hierarchy, priors, sampling design, and validation; avoid interpreting sparse group estimates as stable rankings. |
| Cross-classified mixed models | Outcomes where athletes, coaches, gyms, rulesets, or events overlap rather than nest. | Requires identifiers and adequate replication at each crossed level; no such general model is currently implemented. |
| Physics-informed neural networks (PINNs) | Incorporate defensible biomechanical constraints in pose or motion estimation. | Requires calibrated 3D or otherwise physically meaningful measurements, validated constraints, and suitable labels. Current image-space landmarks alone do not support Newtonian force inference. |
| Topological data analysis (TDA) | Explore shape and persistence features of motion trajectories. | Requires stable, appropriately normalized sequences and comparison against simpler baselines; no combat-specific validation is present. |
| GAMLSS | Model distributional parameters beyond the conditional mean. | Requires an outcome distribution justified by data and enough observations for location/scale/shape estimation. |
| Ornstein-Uhlenbeck process | Test mean-reverting models for a clearly measured continuous state. | No validated energy/fatigue state or fitted OU model exists. Do not equate this proposal with the separate BIOGAS project. |
| Gaussian hidden Markov model (GaussianHMM) | Infer latent states from noisy sequential observations. | Requires suitable sequential features, state-selection diagnostics, and out-of-sample validation. Markov transition tables are not an HMM. |
| Quantum machine learning (QML) | Research-scale comparison of quantum circuits on a bounded task. | No quantum implementation exists. Superposition does not guarantee faster exhaustive search or solve all tournament combinations in milliseconds; benchmark against classical methods. |
| Multi-agent reinforcement learning (MARL) | Simulated policy research in a defined, safe game environment. | No simulator or MARL training loop exists. Simulation results would not establish a real-athlete tactic without external validation. |
| Neuro-symbolic reasoning | Combine learned estimates with explicit, versioned competition rules. | No integrated symbolic rule engine exists. Rules can constrain outputs but cannot guarantee zero hallucinations or replace uncertainty reporting. |
| Federated learning (FL) | Collaborative training across gyms without centralizing raw recordings. | No federated pipeline exists. Requires consent, compatible schemas, secure aggregation, privacy threat analysis, and governance for model updates. Weights alone do not guarantee privacy. |
| Spatial digital twins / AR/VR | Visualize reviewed pose/motion estimates in a spatial interface. | No digital-twin or headset integration exists. Occluded 3D body reconstruction requires appropriate multi-view/depth data and uncertainty handling. |
| Smart-mouthguard impact/concussion models | Research sensor signal quality and externally validated head-impact measurements. | No mouthguard ingestion or medical model exists. Requires device validation, clinical collaboration, outcome data, and regulatory/privacy review; do not infer concussion from generic acceleration. |
| Computational fluid dynamics (CFD) | Research airflow around a modeled athlete or limb. | No CFD pipeline exists. It is not a substitute for measuring impact force or a practical proxy for strike effectiveness without validated boundary conditions. |
| Gaze tracking and cognitive latency | Analyze consented gaze and event-response timing in a controlled protocol. | No eye-tracking integration exists. Requires calibrated hardware, synchronized ground truth, protocol controls, and privacy protections. |
| SMPL / SMPL-X body models | Research 3D body reconstruction or motion representation. | No SMPL/SMPL-X integration exists. Model and source-dataset licenses, registration terms, fit uncertainty, and occlusion errors must be reviewed. |
| Edge AI (TensorRT / ONNX) | Export and benchmark an already validated model on target hardware. | No deployment pipeline or 60-FPS guarantee exists. Requires model export, device-specific profiling, accuracy parity checks, and consent-aware local processing. |

## Architecture and evidence boundaries

- The current Global Quantix Score formula in project materials is a proposed
  specification, not a validated measurement scale. Its component definitions,
  calibration, uncertainty, construct validity, and cross-discipline
  comparability must be established before reporting rankings.
- A model's name or dependency is not evidence that the method has been
  implemented. Only the components in the first table are supported by the
  current code inspection.
- No evidence in this engine establishes an integration with BIOGAS Solvency
  Engine. Any future connection needs an explicit interface, ownership,
  validation, and data-governance review.
- Health-adjacent data and derived outputs require consent, data minimization,
  access controls, and domain-appropriate review. The current telemetry
  summaries are descriptive and deliberately avoid medical conclusions.
- Training and evaluation require rights-cleared data, documented provenance,
  participant consent where applicable, sequence/athlete-aware splits, and
  task-specific metrics. Synthetic demos are not performance evidence.
