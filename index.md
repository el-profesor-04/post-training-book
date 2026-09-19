# Post-Training of Language Models and Agents

## A complete course index and practical study track

**Goal.** Read new post-training papers critically; derive and implement their objectives; build data, reward, rollout, and evaluation pipelines; and extend the same principles to tool-using agents.

**Scope.** Parts 0–XVIII; 75 chapters (three introductory chapters and Chapters 1–72); 25 labs. This is a course map, not a fixed calendar. The mathematical chapters are a deep RL refresher, and the lab track is designed to work with CPU experiments, tiny models, and occasional short GPU sessions.

**Study rhythm.** For each chapter: intuition → mathematics → derivation → numerical example → implementation → paper connection → failure modes → questions → experiment. We can revisit prerequisites whenever they become relevant. Major topics such as PPO, DPO, GRPO, RLVR, reward modeling, and agent credit assignment may take several sessions.

---

## Part 0 — What Post-Training Actually Is

### Chapter 0.1 — The foundation-model lifecycle
Sections: pretraining; mid-training and continued pretraining; instruction tuning; post-training; alignment versus capability elicitation; inference-time versus training-time methods; reasoning models; agents; which interventions change model weights.

### Chapter 0.2 — A taxonomy of post-training
Sections: SFT; preference learning; reward modeling; RLHF; RLAIF; RL with verifiable rewards (RLVR); rejection-sampling fine-tuning; distillation; online versus offline post-training; on-policy versus off-policy learning; self-training; agent post-training.

### Chapter 0.3 — The post-training pipeline
Sections: prompts → rollouts → judgments and verifiers → rewards → advantages → optimization → updated policy → new rollouts. Distinguish data generation, training, and evaluation loops.

## Part I — Mathematical Foundations

### Chapter 1 — Probability for generative models
Sections: autoregressive factorization; conditional probability; log likelihood; cross entropy; KL divergence; forward and reverse KL; entropy; sampling distributions; importance sampling; likelihood ratios.

### Chapter 2 — Language models as policies
Sections: context as state; tokens as actions; sequences as trajectories; terminal reward; token- versus sequence-level reward; deterministic transitions; differences from Atari and robotics; limits of the Markov assumption.

### Chapter 3 — RL fundamentals revisited mathematically
Sections: MDPs; POMDPs; returns; value and Q functions; advantages; Bellman equations; Monte Carlo estimates; temporal difference learning; bias and variance.

### Chapter 4 — Policy gradients
Sections: REINFORCE derivation; log-derivative trick; policy-gradient intuition; Monte Carlo gradients; baselines; variance reduction; actor-critic; advantage estimation; GAE; entropy bonuses.

### Chapter 5 — Trust regions and constrained optimization
Sections: unstable unrestricted updates; KL constraints; TRPO; PPO; clipped surrogate objective; value loss; entropy; adaptive KL penalties; reference policies; distribution shift.

## Part II — Language-Model Training Mechanics

### Chapter 6 — Transformer training at the token level
Sections: tokenization; teacher forcing; causal masks; logits; softmax; token cross entropy; sequence loss; padding masks; prompt masking; completion-only loss.

### Chapter 7 — Optimization
Sections: SGD; Adam and AdamW; momentum; weight decay; learning-rate schedules; warmup; gradient clipping; gradient accumulation; mixed precision; numerical stability.

### Chapter 8 — Parameter-efficient fine-tuning
Sections: LoRA and low-rank updates; rank selection; alpha and scaling; adapter placement; QLoRA; quantization; 8-bit and 4-bit training; adapter merging; full tuning versus PEFT.

### Chapter 9 — Training systems fundamentals
Sections: GPU memory; parameters; gradients; optimizer states; activations; activation checkpointing; data, tensor, pipeline, and sequence parallelism; ZeRO and FSDP.

## Part III — Supervised Post-Training

### Chapter 10 — Instruction tuning and SFT
Sections: instruction datasets; prompt-response formatting; chat templates; multi-turn conversations; assistant-only loss; packing; context length; epochs; learning rates; catastrophic forgetting.

### Chapter 11 — Data quality
Sections: quality versus quantity; diversity; deduplication; contamination; domain balance; curriculum order; difficulty; response style; synthetic data; filtering.

### Chapter 12 — Behavioral cloning
Sections: demonstrations as trajectories; behavior cloning; covariate shift; compounding errors; DAgger; teacher trajectories; successful and failed trajectories; agent trajectory SFT.

### Chapter 13 — Distillation
Sections: hard labels; soft targets; logits; sequence-level distillation; reasoning traces; teacher sampling; on-policy distillation; student-teacher mismatch; reasoning and agent distillation.

## Part IV — Human Preferences and Reward Modeling

### Chapter 14 — Preference data
Sections: pairs; rankings; scalar ratings; accept/reject labels; best-of-N; human and AI annotation; label noise; annotator disagreement; preference ambiguity.

### Chapter 15 — Reward models
Sections: reward-model meaning; Bradley–Terry model; pairwise ranking loss derivation; architecture; sequence classification; normalization; calibration; dataset construction; generalization; uncertainty.

### Chapter 16 — Reward-model pathology
Sections: overoptimization; reward hacking; shift; length, style, and position bias; sycophancy; spurious correlates; Goodhart's law; reward-model ensembles.

### Chapter 17 — Process reward models
Sections: outcome versus process supervision; step judgments; process reward models; step segmentation; credit assignment; Monte Carlo process labels; learned process rewards; hybrids; reward hacking.

## Part V — Classical RLHF

### Chapter 18 — The original RLHF pipeline
Sections: base model; SFT policy; preference collection; reward model; policy optimization; reference model; KL penalty; evaluation loop.

### Chapter 19 — PPO for language models
Sections: rollout, old, and reference policies; critic; reward model; token and terminal rewards; KL-derived token penalties; returns; GAE; PPO clipping; critic loss; entropy; minibatches; PPO epochs.

### Chapter 20 — Why PPO is difficult
Sections: four-model memory footprint; critic instability; hyperparameter coupling; KL instability; reward scale; policy collapse; advantage normalization; stale rollouts; generation/training mismatch; debugging.

## Part VI — Preference Optimization Without Classical RL

### Chapter 21 — DPO from first principles
Sections: KL-regularized RL; optimal policy; implicit reward recovery; Bradley–Terry preferences; DPO derivation; β; chosen/rejected log probabilities; reference model; gradients; failure modes.

### Chapter 22 — The preference-optimization family
Sections: IPO; KTO; ORPO; SimPO; CPO; BCO; NCA; SLiC; RRHF; online DPO; Nash-style methods; reference-free objectives. Compare ideas and assumptions rather than memorizing names.

### Chapter 23 — When preference optimization works
Sections: offline data; coverage; distribution shift; saturation; on-policy data; iterative DPO; self-generated pairs; comparisons with RLHF and SFT; objective selection.

## Part VII — Modern Policy Optimization for LLMs

### Chapter 24 — REINFORCE comes back
Sections: simple gradients in LLM RL; sequence rewards; baselines; leave-one-out baselines; RLOO; ReMax; variance reduction; critic-free RL; comparison with PPO.

### Chapter 25 — GRPO
Sections: group sampling; multiple completions; relative rewards; group normalization; advantage construction; PPO-like clipping; KL; token ratios; critic-free operation; failure modes.

### Chapter 26 — Beyond vanilla GRPO
Sections: Dr.GRPO; REINFORCE++; DAPO; GSPO; sequence- versus token-level ratios; asymmetric clipping; dynamic sampling; length bias; entropy collapse; stability.

### Chapter 27 — Understanding policy-gradient design choices
Sections: sampling, reward, advantage, and importance-ratio units; clipping location; length normalization; prompt weighting; baselines; entropy; actual optimized distribution.

## Part VIII — Reinforcement Learning with Verifiable Rewards

### Chapter 28 — RLVR
Sections: verifiable rewards; math answers; unit tests; code execution; formal proofs; symbolic equivalence; structured output; exact-match pitfalls; partial credit; verifier correctness.

### Chapter 29 — Reasoning emergence
Sections: reasoning traces; long chain-of-thought; search-like reasoning; reflection; backtracking; verification; exploration; strategy discovery; reasoning length; emergent behavior claims and their limits.

### Chapter 30 — Reasoning-model training pipelines
Sections: cold-start SFT; RL stage; rejection sampling; second SFT; general-alignment RL; distillation; trace filtering; readability; general capability preservation; task mixtures.

### Chapter 31 — Reward design for reasoning
Sections: correctness; format; process; length; efficiency; style; anti-hacking; multi-objective rewards; weighting; curricula.

## Part IX — Exploration, Search and Self-Improvement

### Chapter 32 — Exploration in language-model RL
Sections: temperature; top-k and top-p; entropy; diversity; collapse; hard prompts; curriculum learning; difficulty-aware and adaptive sampling; exploration-exploitation tradeoff.

### Chapter 33 — Best-of-N and rejection sampling
Sections: candidate generation; reward ranking; best-of-N; rejection sampling; accepted-generation training; distribution shift; iterative loops.

### Chapter 34 — Search-enhanced training
Sections: beam and tree search; MCTS; value-guided and process-reward search; search-generated supervision; distilling search into a policy; test-time versus train-time compute.

### Chapter 35 — Self-training and self-improvement
Sections: self-generated tasks and answers; self-critique and self-reward; iterative refinement; teacher/student loops; bootstrapping; self-play; curricula; model-collapse risk.

## Part X — Agent Foundations

The setting extends from **prompt → response** to **state → decision → tool/environment → observation → next decision → outcome**.

### Chapter 36 — What makes an LLM agent
Sections: model; harness; tools; environment; memory; control loop; observations; actions; termination; rewards.

### Chapter 37 — Agents as MDPs and POMDPs
Sections: environment state; agent observation; hidden state; tool calls and text as actions; responses; long trajectories; partial observability; history windows; state compression.

### Chapter 38 — Agent architectures
Sections: ReAct; planner/executor; tool-use policies; reflection; critic and verifier agents; hierarchical and multi-agent systems; memory; harnesses.

## Part XI — Agent Post-Training

### Chapter 39 — Agent trajectory datasets
Sections: traces; tool calls and outputs; success/failure; demonstrations; synthetic traces; replay; filtering; compression; action masks.

### Chapter 40 — SFT for tool use
Sections: function-call formats; tool selection; arguments; schemas; multi-tool traces; error recovery; tool results; parallel calls; hallucinated calls; abstention.

### Chapter 41 — RL for agents
Sections: terminal, sparse, dense, environment, and learned rewards; policy gradients; multi-turn and on-policy rollouts; offline trajectories; SFT + RL hybrids.

### Chapter 42 — Credit assignment for long-horizon agents
Sections: trajectory, step, and token credit; temporal assignment; Monte Carlo returns; value functions; process reward models; advantages; counterfactual and hierarchical credit.

### Chapter 43 — Harnessed agentic RL
Sections: runtime versus trainer; decoupled rollouts; API gateways; capturing LLM calls; training sample construction; retokenization; sample merging; weight/version sync; attribution; harness/trainer separation.

### Chapter 44 — Training web and search agents
Sections: search actions; query formulation; results; browsing and navigation; evidence; citations; resets; search rewards; curricula.

### Chapter 45 — Training coding agents
Sections: repository state; shell actions; edits; tests; test and patch rewards; contamination; sandboxes; long trajectories; SWE-bench-style evaluation.

### Chapter 46 — Computer-use agents
Sections: screens; mouse and keyboard; coordinates; vision-language policies; partial observability; long horizons; state aliasing; UI changes; reward construction; safety.

### Chapter 47 — Multi-agent post-training
Sections: cooperation and competition; communication policies; centralized/decentralized training; roles; debate; actor/critic roles; coordination; credit assignment; self-play.

## Part XII — Reward Engineering and Verifiers

### Chapter 48 — Designing good rewards
Sections: proxies; sparse and dense rewards; potential-based shaping; scales and normalization; multi-objective reward; weights; Pareto tradeoffs; reward decomposition.

### Chapter 49 — Verifiers
Sections: rules; unit tests; symbolic verification; LLM judges and model graders; ensembles; pairwise and reference-free grading; calibration; robustness.

### Chapter 50 — Reward hacking
Sections: specification gaming; grader exploits; formatting and length hacks; unit-test gaming; tool/environment exploits; memorized verifier behavior; distributional and adversarial hacking; detection.

## Part XIII — Data Engineering for Post-Training

### Chapter 51 — Prompt distributions
Sections: training versus production distributions; difficulty; long tails; sampling weights; domain mixtures; dynamic curricula; hard-example mining.

### Chapter 52 — Synthetic data generation
Sections: teachers; prompt, response, preference, critique, and task synthesis; difficulty control; filtering; diversity; synthetic-data collapse.

### Chapter 53 — Data flywheels
Sections: production interactions; failure mining; human correction; grader labels; retraining; re-evaluation; continuous improvement; online learning concerns.

## Part XIV — Evaluation Science

### Chapter 54 — Offline evaluation
Sections: accuracy; exact match; pass@k; reward; win rates; pairwise evaluation; calibration; confidence intervals; bootstrap; statistical significance.

### Chapter 55 — LLM-as-a-judge
Sections: prompting; pairwise and reference-based judging; position and verbosity bias; self-preference; calibration; human agreement; ensembles; adversarial evaluation.

### Chapter 56 — Agent evaluation
Sections: success; steps; tool calls; cost; latency; error recovery; robustness; environmental variance; trajectory quality; success-efficiency tradeoffs.

### Chapter 57 — Benchmark pathology
Sections: contamination; overfitting; gaming; saturation; hidden tests; distribution shifts; validity; private evaluation suites.

### Chapter 58 — Experimental design
Sections: baselines; ablations; seeds; sample sizes; learning curves; sweeps; compute- and data-matched comparisons; statistics; reproducibility.

## Part XV — Post-Training Systems Engineering

### Chapter 59 — Rollout infrastructure
Sections: training and inference workers; serving engines; batched and continuous generation; throughput; KV cache; variable lengths; long contexts; rollout scheduling.

### Chapter 60 — Training/inference disaggregation
Sections: actor, rollout, reward, and critic workers; weight synchronization and broadcast; stale policies; colocation; disaggregation; asynchronous RL.

### Chapter 61 — Distributed post-training
Sections: FSDP; ZeRO; tensor, pipeline, sequence, and expert parallelism; MoE training; communication bottlenecks; checkpoints; fault recovery.

### Chapter 62 — RL-specific systems problems
Sections: sampling/training imbalance; utilization; stragglers; variable rollout lengths; staleness; replay; tokenization drift; numerical mismatch; reproducibility; observability.

## Part XVI — Safety and Alignment Post-Training

### Chapter 63 — Helpful, honest, and safe behavior
Sections: behavioral alignment; refusal and over-refusal; sycophancy; truthfulness; uncertainty; instruction hierarchy; conflicting objectives.

### Chapter 64 — RLAIF and Constitutional AI
Sections: AI feedback; principles; self-critique; response revision; AI preferences; reward modeling; constitutional RL; scalable supervision.

### Chapter 65 — Adversarial post-training
Sections: red teaming; jailbreak data; adversarial SFT and preference learning; reward-model attacks; distribution shifts; safety regression; agent risks.

### Chapter 66 — Alignment tradeoffs
Sections: capability and safety; helpfulness and harmlessness; reward conflict; refusal boundaries; pluralistic preferences; personalization; model specifications and constitutions.

## Part XVII — Specialized and Frontier Topics

### Chapter 67 — Multimodal post-training
Sections: vision-language SFT; preference learning; multimodal rewards; grounding; vision-agent RL; computer use; video agents; multimodal reasoning.

### Chapter 68 — Long-context post-training
Sections: sequence-length scaling; context curricula; long-context SFT; long-rollout RL; credit assignment; memory; compression; very long sequence training.

### Chapter 69 — Tool and retrieval specialization
Sections: retrieval policies; query rewriting; retrieval rewards; RAG post-training; tool choice and sequencing; tool abstention; cost-aware agents.

### Chapter 70 — Continual and online learning
Sections: continual SFT and preferences; online RL; nonstationarity; forgetting; replay; distribution monitoring; production feedback.

### Chapter 71 — Meta-learning and automatic curriculum generation
Sections: task generators; difficulty predictors; adaptive curricula; self-play; adversarial tasks; learning progress; automatic environment generation.

### Chapter 72 — Open research problems
Sections: long-horizon credit; sparse rewards; oversight; process rewards; exploration; reward hacking; RL generalization; stable off-policy LLM RL; continual agents; self-improvement; multi-agent learning; production feedback; alignment under capability growth; compute efficiency.

## Part XVIII — No-GPU Experimental Track

Each lab has a CPU/toy version. Larger models or longer rollouts are optional extensions, not prerequisites.

1. **Tiny autoregressive model.** Train a causal LM and inspect token cross entropy.
2. **Tiny instruction SFT.** Format examples and tune a small instruction model.
3. **LoRA from scratch.** Implement and inspect low-rank update matrices.
4. **Preference dataset.** Create chosen/rejected pairs and measure label noise.
5. **Reward model.** Implement and train Bradley–Terry pairwise loss.
6. **REINFORCE for text.** Train on an artificial sequence task.
7. **KL-regularized RLHF.** Implement the penalty and inspect its effect.
8. **PPO.** Implement a small policy and inspect every important tensor.
9. **DPO.** Derive the loss, implement it, and compare with SFT.
10. **KTO/SimPO-style objectives.** Change the objective and compare gradients.
11. **RLOO.** Sample several responses and use leave-one-out baselines.
12. **GRPO.** Train on a tiny automatically checked math task.
13. **Algorithm comparison.** Hold model, prompts, and rewards fixed across GRPO, RLOO, and REINFORCE.
14. **Verifier.** Build a math or code correctness checker.
15. **Miniature RLVR.** Improve measured performance on a verifiable task.
16. **Reward hacking.** Introduce a flawed verifier and document the exploit.
17. **Process reward model.** Learn step-level judgments on short reasoning traces.
18. **Tool-use environment.** Connect an LM to a calculator, search index, or database and feed observations back.
19. **Agent trajectory collection.** Record states, actions, tool results, and outcomes.
20. **Agent behavior cloning.** Learn from successful trajectories.
21. **Agent RL.** Optimize for terminal task success.
22. **Credit assignment study.** Compare terminal rewards with shaping and process rewards.
23. **Coding-agent environment.** Give the agent tiny repositories and tests.
24. **Test-based coding-agent RL.** Use test results as the verifier.
25. **Miniature post-training system.** Separate rollout server → verifier → trajectory store → trainer → evaluation server.

### Compute tiers

- **A — CPU/laptop:** toy policies, small neural nets, RL environments, analytical and numerical derivations, and tiny transformers.
- **B — small language models:** roughly 100M–1B parameters, with adapters or quantization where feasible; exact feasibility depends on the machine.
- **C — occasional GPU:** a short free or rented session for experiments that genuinely need acceleration.
- **D — frontier-systems simulation:** reproduce mechanisms with small models, inspect published systems, and estimate memory, throughput, and scale effects.

### End-of-course standard

For a new method, identify its sampling distribution, reward, baseline, advantage, importance ratio, KL treatment, gradient estimator, failure modes, implementation details, rollout infrastructure, and fair evaluation. For an agent, specify the harness-induced MDP/POMDP, decision boundaries, trajectory data, credit assignment, verifier, and training loss.
