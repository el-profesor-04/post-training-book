# Post-Training of Language Models and Agents

## A course in research and research engineering

**Edition 1 · September 2026**

**Audience:** a programmer with some machine learning and reinforcement learning experience who wants to derive, implement, evaluate, and investigate post-training methods.

**Course map:** 20 parts, 81 chapters (0.1–0.3 and 1–78), four mathematical modules, and 30 integrated labs.

---

## How to study this course

Read with a notebook and a Python interpreter. For each important equation, identify the random variables, their sampling distribution, what is held fixed, and what receives gradients. Reproduce a small numerical example before trusting a large training run. When an experiment disagrees with a derivation, investigate both.

The research engineering goal is to implement and debug a complete post-training pipeline. The research goal is to explain why it behaves as it does, isolate competing explanations, and design experiments capable of rejecting a hypothesis. The same experiment can serve both goals.

This manuscript contains the teaching material, worked examples, practical assignments, and research prompts for the full index. Paper readings deepen the lessons. Frontier topics are taught as research areas with explicit limitations, not as settled recipes. A toy experiment can test an estimator identity or expose a failure mechanism; it cannot establish that a recipe will scale to a frontier model.

### Study cycle

1. State the problem and predict the outcome of the example.
2. Work through the explanation and derivation.
3. Recompute the numerical example without looking.
4. Implement or inspect the relevant tensors.
5. Complete the checkpoint and any lab scheduled there.
6. Record what failed, what changed, and what remains uncertain.
7. Read the linked primary source and compare its assumptions with this lesson.

Keep a research log with columns for hypothesis, intervention, controlled variables, measurements, result, and revised belief. Passing a checkpoint means being able to explain the answer, not merely recognizing a term.

### Compute tiers

| Tag | Meaning | What it teaches |
|---|---|---|
| **A** | CPU/laptop: finite policies, small networks, tiny transformers, local environments | Objectives, gradients, data pipelines, controlled experiments |
| **B** | Optional small language model, roughly 100M–1B parameters if memory permits | Tokenization, realistic distributions, adapter training |
| **C** | Optional short GPU session | Faster experiments and larger batches or models |
| **D** | Systems simulation with small models or mock workers | Scheduling, synchronization, scaling estimates, failure recovery |

These tags describe experiment forms, not a ladder of competence. Parameter count alone does not determine whether a model fits: context length, activations, optimizer state, precision, and batching also matter. Every required lab has an A route. For B/C, choose a model only after measuring memory and throughput on a short pilot.

### Lab conventions

Lab IDs match the index. They are not consecutive in reading order because some labs need later material. Each lab specifies an objective, procedure, evidence to submit, and checks. “Expected behavior” means a prediction to test, never fabricated experimental results. No claimed learning curve or benchmark gain in this manuscript represents an experiment run for you.

The default experiment is a finite answer policy or a tiny character model. Appendix A supplies a runnable CPU kernel for exact-gradient and estimator experiments. Later labs extend that kernel or the tiny language model. You will write the remaining training loops as part of the work.

### Mathematical conventions

| Symbol | Meaning |
|---|---|
| $x\sim D$ | Prompt or task drawn from a specified distribution |
| $y=(y_1,\ldots,y_T)$ | Generated response, including termination when modeled |
| $h_t=(x,y_{<t})$ | History before response token $t$ |
| $\pi_\theta$ | Trainable policy |
| $\mu$, $\pi_{\mathrm{old}}$ | Behavior policy that generated a stored rollout |
| $\pi_{\mathrm{ref}}$ | Reference policy used for regularization |
| $R(x,y)$ | Whole-response reward |
| $r_t$ | Reward at a particular decision |
| $V^\pi,Q^\pi,A^\pi$ | Value, action value, and advantage |
| $\gamma,\lambda$ | Discount and GAE trace parameter |
| $\beta$ | KL regularization coefficient unless a local definition says otherwise |
| $\rho_t$ | Current/behavior action-probability ratio |
| $\text{sg}(\cdot)$ | Stop gradient: treat an estimated quantity as fixed during this update |

Unless stated otherwise, expectations and logarithms use natural logarithms, prompts are sampled independently of policy parameters, and finite-horizon text tasks use $\gamma=1$. For infinite-horizon statements, $0\leq\gamma<1$. A positive objective $J$ is maximized; a loss $L$ is minimized.

The recurring objective is

$$
J_\beta(\theta)=
\mathbb E_{x\sim D}
\left[
\mathbb E_{y\sim\pi_\theta(\cdot\mid x)}R(x,y)
-\beta D_{\mathrm{KL}}\left(
\pi_\theta(\cdot\mid x)\,\|\,\pi_{\mathrm{ref}}(\cdot\mid x)
\right)
\right].
$$

This is a distribution-level definition. A particular trainer may optimize an approximation or surrogate. We will identify that distinction every time it matters.

### Figure placeholders

Figure callouts specify a stable ID, suggested content, and a caption. They deliberately contain no broken image links. Equations, tables, and explanations remain usable before the illustrations are added. Appendix C collects the figure specifications.

## Course plan and navigation

| Part | Chapters | Main outcome |
|---|---|---|
| [0 · Orientation](#part-0) | 0.1–0.3 | Trace a complete post-training intervention |
| [I · Mathematical foundations](#part-i) | 1–5, M1–M4 | Derive the estimators and optimization constraints |
| [II · Training mechanics](#part-ii) | 6–9 | Build and inspect a tiny language-model trainer |
| [III · Supervised post-training](#part-iii) | 10–13 | Construct SFT, cloning, and distillation pipelines |
| [IV · Preferences and rewards](#part-iv) | 14–17 | Model feedback and diagnose reward errors |
| [V · Classical RLHF](#part-v) | 18–20 | Implement and debug PPO with KL regularization |
| [VI · Preference optimization](#part-vi) | 21–23 | Derive DPO and compare alternative objectives |
| [VII · Modern policy optimization](#part-vii) | 24–27 | Analyze RLOO, GRPO, and estimator design |
| [VIII · Verifiable rewards](#part-viii) | 28–31 | Build a verified learning task and evaluate reasoning |
| [IX · Search and self-improvement](#part-ix) | 32–35 | Understand exploration and generated supervision |
| [X · Agent foundations](#part-x) | 36–38 | Specify a tool-using policy and its environment |
| [XI · Agent post-training](#part-xi) | 39–47 | Train and evaluate multi-step agent behavior |
| [XII · Reward engineering](#part-xii) | 48–50 | Shape rewards and audit verifiers |
| [XIII · Data engineering](#part-xiii) | 51–53 | Control distributions and feedback loops |
| [XIV · Evaluation science](#part-xiv) | 54–58 | Make defensible empirical comparisons |
| [XV · Systems engineering](#part-xv) | 59–62 | Build a miniature distributed training system |
| [XVI · Safety and alignment](#part-xvi) | 63–66 | Evaluate behavior under conflicting objectives |
| [XVII · Specialized topics](#part-xvii) | 67–72 | Transfer the framework to frontier settings |
| [XVIII · Experimental pathway](#part-xviii) | Lab synthesis | Assemble the practical portfolio |
| [XIX · Doing research](#part-xix) | 73–78 | Produce a small, rigorous research investigation |
| [Appendices](#appendices) | A–D | CPU kernel, solutions, figure register, references |

The writing and study plan follows these parts in order. Within each part, establish the objects first, derive or analyze the mechanism second, and test it third. Mathematical modules are reference lessons: use the indicated module when a derivation needs it, then return to the chapter.

---

<a id="part-0"></a>
## Part 0 — What Post-Training Actually Is

**Chapter navigation:** [0.1 · The foundation-model lifecycle](#chapter-0-1) · [0.2 · A taxonomy of post-training](#chapter-0-2) · [0.3 · The post-training pipeline](#chapter-0-3)

**Learning goals:** classify an intervention, distinguish training from deployment behavior, and name every distribution in a training loop.

<a id="chapter-0-1"></a>
### Chapter 0.1 — The foundation-model lifecycle

#### 0.1.1 From next-token prediction to useful behavior

Pretraining estimates patterns in a broad data distribution using a next-token objective. It produces a model with many possible continuations, not a guarantee that the continuation a user wants receives the highest probability. A language model may know how to write both a correct explanation and a plausible incorrect one.

Continued pretraining, sometimes called mid-training when it forms an intermediate stage, exposes a model to additional domain or capability data. The labels are usage conventions, not a mathematical boundary: identify the objective, data distribution, and parameter update. Training on scientific documents with next-token loss is continued pretraining even if it happens late.

Instruction tuning uses demonstrations of desired responses. Preference optimization teaches distinctions between responses. Reinforcement learning uses rewards attached to sampled behavior. These can improve task performance, select existing behaviors, learn new strategies, or alter response style. An observed gain alone does not tell you which mechanism occurred.

#### 0.1.2 Capability, alignment, and the deployed system

Capability concerns what tasks a system can perform under specified resources. Alignment concerns whether its behavior matches intended requirements. They overlap: following a tool schema can be both a capability and a behavioral requirement. A refusal policy can lower measured success on one test while improving compliance with an intended boundary.

A reasoning model is typically encouraged or trained to spend computation on intermediate reasoning. An agent repeatedly selects actions in an environment. Neither label uniquely identifies a loss function.

| Intervention | Usually changes weights? | Immediate mechanism |
|---|---:|---|
| Continued pretraining | Yes | Predict additional corpus tokens |
| SFT or RL | Yes | Update the policy using demonstrations or rewards |
| A better prompt or retrieval index | No | Change the context presented to the policy |
| Best-of-N at inference | No | Select among sampled responses |
| Distilling best-of-N outputs | Yes | Train on selected responses |
| Changing an agent's tool permissions | No | Change the available actions and transitions |

**Worked example.** A coding assistant improves from 40 to 55 successes on a fixed test suite. You simultaneously add repository search and train on successful edits. You cannot attribute the gain to training without evaluating the four combinations of old/new weights and old/new search.

**Checkpoint.** Classify adding a calculator, SFT on calculator traces, and using a calculator to verify training outputs. They respectively alter the environment, the policy, and the learning signal.

<a id="chapter-0-2"></a>
### Chapter 0.2 — A taxonomy of post-training

#### 0.2.1 Supervision and collection are separate choices

SFT minimizes negative log likelihood of demonstrations. Reward modeling fits a predictor of quality or preference. RLHF uses human-derived feedback in a reinforcement-learning pipeline; RLAIF substitutes or supplements AI feedback. RLVR uses an executable correctness signal such as a test or proof checker. A verifier can still be incomplete or wrong.

Rejection-sampling fine-tuning generates candidates, retains acceptable ones, then applies supervised learning. Distillation transfers behavior from a teacher, which may be a model, search procedure, or ensemble. Self-training reuses model-generated data; self-improvement is a claim about the resulting changes and must be evaluated independently.

| Axis | First setting | Second setting | Question to ask |
|---|---|---|---|
| Data refresh | Offline: fixed dataset | Online: newly collected data | Does the collection distribution change? |
| RL sampling | On-policy | Off-policy | Did the current policy generate this behavior? |
| Feedback | Outcome-level | Process-level | Is only the endpoint evaluated? |
| Interaction | Single response | Multi-step agent | Can actions change future observations? |
| Policy update | Supervised likelihood | Reward-based optimization | What quantity supplies the gradient? |

Online and on-policy are not synonyms. A continuously collected dataset can contain trajectories from stale policies. A single freshly generated rollout becomes off-policy relative to sufficiently changed weights.

#### 0.2.2 A useful classification exercise

Suppose yesterday's model generates two answers, an AI judge chooses one, and today's model receives DPO updates on the stored pairs. This is preference optimization on offline pairs for that update; the wider process can be an online collection loop. It uses AI feedback but does not require PPO or a separately trained scalar reward model.

**Checkpoint.** Explain why “we use RLHF” is insufficient to reproduce a method. At minimum you need collection policy, prompts, reward source, estimator, optimizer, regularization, and evaluation.

<a id="chapter-0-3"></a>
### Chapter 0.3 — The post-training pipeline

#### 0.3.1 Three loops and their contracts

The data loop chooses tasks, generates responses, obtains labels, and maintains datasets. The training loop constructs rewards and advantages, calculates gradients, and updates weights. The evaluation loop measures frozen checkpoints on held-out tasks. Evaluation data must not silently become training data.

Every rollout should identify prompt ID, policy version, decoding settings, tokens, token log probabilities where needed, termination reason, reward version, and environment version. For an agent, add actions, observations, tool results, and timestamps. This record is the evidence needed to reconstruct the update.

#### 0.3.2 One example from beginning to end

Take a prompt asking for a two-digit sum. The policy samples a response. A parser extracts exactly one integer. A checker compares it with the sum, producing reward 0 or 1. A baseline estimates expected reward for that prompt. The advantage measures whether the sample did better than expected. The optimizer increases or decreases the probability of the sampled response, with regularization if specified.

The reward is not the gradient. The gradient also depends on the probability model, baseline, token mask, normalization, and sampling distribution. Two systems can use the same reward and optimize materially different objectives.

> **Figure placeholder F01 — The three-loop post-training pipeline.** Draw data collection, policy updates, and held-out evaluation as separate feedback loops. Label policy, dataset, reward, and environment versions at their interfaces. Caption: “A reproducible experiment records every object that can change behavior.”

**Exercise.** Write one JSON record for an arithmetic rollout and identify which fields are required by SFT, DPO, and PPO. PPO additionally needs the probabilities of the behavior policy; DPO needs paired responses and reference scores, which may be recomputed.

**Part 0 deliverable:** a one-page specification of the task, policy, environment, feedback, update, and evaluation for your first experiment.

<a id="part-i"></a>
## Part I — Mathematical Foundations

**Chapter navigation:** [1 · Probability for generative models](#chapter-1) · [2 · Language models as policies](#chapter-2) · [3 · RL fundamentals revisited mathematically](#chapter-3) · [4 · Policy gradients](#chapter-4) · [5 · Trust regions and constrained optimization](#chapter-5)

**Learning goals:** derive a policy gradient, explain the role of baselines and KL constraints, and recognize when a sampled loss is only a surrogate for an objective.

**Mathematical support:** use M1 for differentiation, M2 for expectations, M3 for KL, and M4 for constrained optimization.

<a id="chapter-1"></a>
### Chapter 1 — Probability for generative models

#### 1.1 A response is a product of conditional choices

An autoregressive model assigns probability

$$
\pi_\theta(y \mid x) = \prod_{t=1}^{T} \pi_\theta(y_t \mid x, y_{< t}),
\qquad
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta(y_t \mid x, y_{< t})
$$

This follows from the probability chain rule; it does not assert that tokens are independent. The history includes all earlier tokens. Include the end-of-sequence event when comparing probabilities of complete variable-length responses. Otherwise the score describes a prefix.

**Example.** If three sampled tokens have probabilities $0.5,0.2,0.8$, the sequence probability is $0.08$ and its log probability is approximately $-2.526$. Averaging token log probabilities gives $-0.842$; this is a length-normalized score, not the log probability of the sequence.

Teacher-forced maximum likelihood minimizes cross entropy against a data distribution $q$:

$$
H(q,\pi_\theta)=-\mathbb E_{y\sim q}\log\pi_\theta(y).
$$

Since $H(q,\pi_\theta)=H(q)+D_{\mathrm{KL}}(q\|\pi_\theta)$, minimizing cross entropy projects the data distribution toward the model family in that KL direction. The data entropy does not depend on $\theta$.

#### 1.2 Entropy and KL direction

Entropy $H(p)=-\sum_y p(y)\log p(y)$ measures uncertainty under $p$. KL is

$$
D_{\mathrm{KL}}(p\|q)=\sum_y p(y)\log\frac{p(y)}{q(y)}.
$$

It is nonnegative and generally asymmetric. “Forward” and “reverse” are ambiguous unless both arguments are named. In this course, the standard RLHF penalty is policy-to-reference $D_{\mathrm{KL}}(\pi_\theta\|\pi_{\mathrm{ref}})$.

For $p=(0.8,0.2)$ and $q=(0.5,0.5)$, $D_{\mathrm{KL}}(p\|q)\approx0.193$, while $D_{\mathrm{KL}}(q\|p)\approx0.223$. Neither number is a distance in the metric sense.

The informal descriptions “mode covering” and “mode seeking” concern particular approximation settings. They are not universal guarantees about a neural optimizer.

#### 1.3 Importance sampling and support

If $y\sim\mu$, then

$$
\mathbb E_{y\sim\pi}f(y)
=\mathbb E_{y\sim\mu}
\left[\frac{\pi(y)}{\mu(y)}f(y)\right],
$$

provided $\mu(y)>0$ wherever $\pi(y)f(y)\neq0$, and the expectation exists. This support condition matters when top-k sampling sets probabilities to zero.

For an autoregressive response, the full ratio is a product of token ratios, or equivalently the exponential of a sum of log-ratio differences. Long products can have enormous variance. Replacing the full ratio with a token ratio is not a free algebraic simplification: it usually defines a different approximation or local policy surrogate.

**Checkpoint.** If a sampler uses temperature 0.7 but you save log probabilities at temperature 1, which policy appears in the importance-sampling denominator? The actual sampler. A mismatch corrupts the correction.

#### 1.4 Derivation: the sequence KL chain rule

The sequence-level KL is not a sum of token KLs at arbitrary prefixes. Its prefix distribution matters. Starting from autoregressive factorization,

$$
D_{\mathrm{KL}}(\pi\|q)
=\mathbb E_{y\sim\pi}\sum_t
\log\frac{\pi(y_t\mid h_t)}{q(y_t\mid h_t)}.
$$

Condition on the prefix before taking the next-token expectation:

$$
D_{\mathrm{KL}}(\pi\|q)=
\sum_t\mathbb E_{h_t\sim d_t^\pi}
D_{\mathrm{KL}}\bigl(\pi(\cdot\mid h_t)\|q(\cdot\mid h_t)\bigr).
$$

For variable length, use EOS termination or an absorbing state so the sum is well-defined. The identity includes both how often a prefix is visited and how the next-token distributions differ there.

This has a practical implication. Computing an exact vocabulary KL at prefixes drawn from an old policy does not by itself compute the new policy's exact sequence KL. The token expectation is exact at those prefixes, but the prefix distribution is still old.

**Research check:** compare two policies that agree at the initial token but diverge strongly after a rare prefix. A small average KL on an evaluation sample can miss the rare branch. Report tail behavior or targeted prefix probes when the branch matters.

<a id="chapter-2"></a>
### Chapter 2 — Language models as policies

#### 2.1 States, actions, and rewards

For a single response, let state $s_t=h_t=(x,y_{<t})$, action $a_t=y_t$, and transition be appending the token. The transition is deterministic, while the policy is stochastic. Randomness in task choice and sampling still makes the overall experiment stochastic.

A terminal reward arrives after the completed response. Token-level rewards can penalize length, encourage some local behavior, or encode a regularizer. If the desired utility concerns the whole answer, assigning it to every token without specifying the return calculation can multiply its effective weight.

| Object | Single response | Tool-using agent |
|---|---|---|
| Decision | Next token | Next token or structured action |
| State transition | Append token | Execute action; update world and context |
| Observation | Accumulated text | Tool result, screen, retrieved document |
| Termination | EOS or length limit | Success, failure, budget, or explicit stop |
| Hidden state | Usually abstracted away | Often substantial |

#### 2.2 Where the analogy breaks

Atari and robotics often use repeated interaction with a dynamic world. A text-completion environment can be almost entirely known, yet its action space and horizon are large. The Markov property holds for the complete relevant history under a specified task generator; it need not hold for a truncated context or a lossy memory summary.

The environment's terminal checker may depend on hidden information, such as a private test. A policy can still use the observable history, but its input is not the full environment state. That leads naturally to a POMDP.

**Worked example.** A two-token binary response has four possible trajectories: 00, 01, 10, and 11. A reward of 1 for 11 induces credit to both token choices. If only the second token were trained, the policy might learn the final choice while preserving a poor probability of reaching the right prefix.

**Exercise.** Specify which variables must be included in state for a coding environment: repository content, working directory, pending processes, tool outputs, and remaining budget are candidates. Explain what is hidden from the model.

<a id="chapter-3"></a>
### Chapter 3 — RL fundamentals revisited mathematically

#### 3.1 Returns and values

An MDP is a state space, action space, transition law, reward law, initial-state distribution, and discount. Define

$$
G_t=\sum_{k=t}^{T-1}\gamma^{k-t}r_k,\quad
V^\pi(s)=\mathbb E_\pi[G_t\mid s_t=s],\quad
Q^\pi(s,a)=\mathbb E_\pi[G_t\mid s_t=s,a_t=a].
$$

Then $A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)$. Advantage measures relative performance at a state, not absolute task quality.

The Bellman expectation equations are

$$
V^\pi(s)=\mathbb E_{a\sim\pi}[Q^\pi(s,a)],
\qquad
Q^\pi(s,a)=\mathbb E[r+\gamma V^\pi(s')\mid s,a].
$$

These identities decompose the return into immediate reward and future value. They do not require that an approximate learned value function satisfy them exactly.

#### 3.2 Monte Carlo and temporal difference estimates

Monte Carlo uses a sampled complete return as a target. A one-step TD target is $r_t+\gamma V_\phi(s_{t+1})$. Monte Carlo can have high variance; TD can introduce bias through an inaccurate bootstrap.

For rewards $(0,0,1)$ and $\gamma=1$, all three returns equal 1. With $\gamma=0.9$, they are $0.81,0.9,1$. Discounting therefore changes how the system values delays. In a reasoning task, it can introduce pressure toward shorter successful responses.

Distinguish terminal states from truncation. At a true terminal state, future value is zero. If an episode is cut off only because the data collector reached a time limit, bootstrapping may be appropriate for the underlying continuing task. If the budget is itself part of the task definition, exhausting it can be a genuine terminal failure.

#### 3.3 Occupancy and partial observability

For an infinite discounted problem, define normalized state occupancy

$$
d^\pi(s)=(1-\gamma)\sum_{t=0}^{\infty}\gamma^t
\Pr_\pi(s_t=s).
$$

Changing the policy changes both action probabilities and the states encountered. This is why supervised imitation on expert states can fail when the learner drifts.

In a POMDP, a belief state is a posterior over hidden states given history. Keeping full history or a learned memory is a practical approximation to maintaining that belief.

**Checkpoint.** Is a negative advantage necessarily a bad action? No. A reward of 0.9 can be below a state baseline of 0.95.

#### 3.4 Solve a value function instead of naming it

Consider a state $s_0$ with two actions. Action A terminates with reward 0.4. Action B moves with zero immediate reward to $s_1$. At $s_1$, action C succeeds for reward 1 and action D fails for 0. Let $\pi(B\mid s_0)=0.5$, $\pi(C\mid s_1)=0.8$, and $\gamma=1$.

Then $V(s_1)=0.8$, $Q(s_0,A)=0.4$, and $Q(s_0,B)=0.8$. Therefore $V(s_0)=0.5(0.4)+0.5(0.8)=0.6$. Advantages at $s_0$ are $-0.2$ for A and $+0.2$ for B.

Now weaken the second-stage policy so $\pi(C\mid s_1)=0.2$. The value of B falls to 0.2 and A becomes the better action under the current policy. This illustrates a central point: action values include the quality of future decisions, not just the environment's theoretical possibility of success.

For an agent, a powerful tool can have low value under a policy that cannot interpret its output. A curriculum or demonstration can improve the later decision and thereby make earlier information-gathering actions worth selecting.

<a id="chapter-4"></a>
### Chapter 4 — Policy gradients

#### 4.1 Derivation from a finite distribution

Start with $J(\theta)=\sum_y\pi_\theta(y)R(y)$, with reward independent of $\theta$. Differentiate:

$$
\nabla_\theta J
=\sum_y R(y)\nabla_\theta\pi_\theta(y)
=\sum_y\pi_\theta(y)R(y)\nabla_\theta\log\pi_\theta(y).
$$

Thus

$$
\nabla_\theta J=
\mathbb E_{\pi_\theta}
\left[R(y)\sum_t\nabla_\theta\log\pi_\theta(y_t\mid h_t)\right].
$$

The log-derivative trick converts a derivative of a distribution into a sampled score function. Interchanging differentiation and integration in continuous or infinite settings requires regularity conditions; the finite setting makes the algebra transparent.

In an MDP, environment transition factors disappear from the derivative when they do not depend on $\theta$. Rewards before an action can be omitted from its learning signal by conditioning: the expected score of that action is zero. This gives the reward-to-go form. For the objective $\mathbb E\sum_t\gamma^t r_t$, the finite-trajectory gradient includes $\gamma^t G_t$; discounted occupancy formulations absorb this weighting differently.

#### 4.2 Why baselines work

For any baseline $b(s)$ that does not depend on the sampled action conditional on state,

$$
\mathbb E_{a\sim\pi_\theta(\cdot\mid s)}
[b(s)\nabla_\theta\log\pi_\theta(a\mid s)]
=b(s)\nabla_\theta\sum_a\pi_\theta(a\mid s)=0.
$$

Subtracting it does not change the expected gradient. It can change variance dramatically. An action-dependent baseline generally requires an additional correction.

Use stop gradient on a learned baseline inside the actor loss. Otherwise the optimizer also differentiates through the baseline, which is not the policy-gradient estimator just derived.

**Numerical example.** Let $\pi(1)=\sigma(z)=p$, with rewards $R(1)=1,R(0)=0$. Then $J=p$, so $dJ/dz=p(1-p)$. At $p=0.25$, the exact derivative is $0.1875$. A sampled estimator $R(a)(a-p)$ equals 0.75 when action 1 occurs and 0 otherwise. Its expectation is $0.25(0.75)=0.1875$. This is the simplest oracle for checking a trainer.

#### 4.3 Actor-critic and GAE

A critic estimates value. The actor uses an advantage estimate rather than the raw return. Define TD residual

$$
\delta_t=r_t+\gamma(1-d_t)V_\phi(s_{t+1})-V_\phi(s_t),
$$

where $d_t=1$ denotes true termination. Generalized advantage estimation uses

$$
\hat A_t^{\mathrm{GAE}}
=\sum_{l\ge0}(\gamma\lambda)^l
\left(\prod_{j=0}^{l-1}(1-d_{t+j})\right)\delta_{t+l}.
$$

Compute it backward:

$$
\hat A_t=\delta_t+\gamma\lambda(1-d_t)\hat A_{t+1}.
$$

With $\lambda=0$, this is one-step TD. With $\lambda=1$ and complete episodes, it telescopes to Monte Carlo return minus value. Intermediate values trade reliance on noisy samples against reliance on the critic. The detailed bias discussion also depends on the discount and target objective. See [Generalized Advantage Estimation][gae].

An entropy bonus encourages stochasticity. For a categorical distribution, compute $-\sum_a\pi(a)\log\pi(a)$ directly when feasible. Entropy in irrelevant wording is not necessarily exploration of useful strategies.

**Exercise.** For $\gamma=1,\lambda=0.5$, TD residuals $0.2,-0.1,0.6$ give advantages $0.3,0.2,0.6$. Verify using backward recursion.

#### 4.4 Derivation: why past rewards disappear

The trajectory score is $\sum_t\nabla\log\pi(a_t\mid h_t)$. Multiplying it by the total return initially assigns every reward to every action. For a reward $r_k$ occurring before action $a_t$, $k<t$, condition on $h_t$. The past reward is already fixed, while

$$
\mathbb E[\nabla\log\pi(a_t\mid h_t)\mid h_t]=0.
$$

Thus $\mathbb E[r_k\nabla\log\pi(a_t\mid h_t)]=0$. Removing such terms preserves the expected gradient and usually reduces variance. The remaining signal is reward-to-go.

This is a statement about temporal order and conditional independence. It does not mean an early action has no influence on a later reward; those future terms remain precisely because it can.

For a tool agent, a reward issued after a search call but before the next call cannot be caused by that next call. Attaching it to later action tokens adds noise. Conversely, a final task outcome can provide credit to all earlier actions.

**Implementation exercise:** create a two-step task where the first reward depends only on the first action. Compare variance with and without including that reward in the second action's score term. The expected gradient should agree.

<a id="chapter-5"></a>
### Chapter 5 — Trust regions and constrained optimization

#### 5.1 Why a locally good gradient can fail globally

A gradient describes local change. A large update can shift the policy into states where a value model or reward model is inaccurate. A KL constraint limits distribution change in a specified direction and under a specified state distribution.

For discounted MDPs, the performance-difference identity is

$$
J(\pi')-J(\pi)
=\frac{1}{1-\gamma}
\mathbb E_{s\sim d^{\pi'},a\sim\pi'}[A^\pi(s,a)].
$$

The new occupancy $d^{\pi'}$ is inconvenient. A local surrogate uses old-policy states instead. Trust-region reasoning controls how much that approximation can change. The exact monotonic-improvement arguments use assumptions and bounds that practical finite-batch implementations only approximate. See [Trust Region Policy Optimization][trpo].

#### 5.2 TRPO, PPO, and two different references

TRPO optimizes an advantage surrogate with an old-to-new average KL constraint, using curvature information and a line search in its practical algorithm. PPO simplifies optimization using a clipped surrogate:

$$
J_{\mathrm{clip}}(\theta)=
\mathbb E_{\mathrm{old}}\left[
\min\left(
\rho_t\hat A_t,
\text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t
\right)\right],
\quad
\rho_t=\frac{\pi_\theta(a_t\mid s_t)}
{\pi_{\mathrm{old}}(a_t\mid s_t)}.
$$

Clipping removes incentives for some overly large ratio changes. It is not a hard constraint on every probability or a guarantee that KL remains below a threshold. The old policy is fixed for a rollout/update cycle. The reference policy used by RLHF can remain fixed across many cycles.

For advantage $+2$, ratio 1.4, and $\epsilon=0.2$, the clipped term is 2.4 instead of 2.8. For advantage $-2$ and ratio 0.6, the minimum is $-1.6$, preventing unlimited benefit from reducing the probability of that sampled action.

#### 5.3 Penalties and adaptive control

A constrained problem $\max_\pi \mathbb E R$ subject to KL $\leq\kappa$ has a Lagrangian with nonnegative multiplier $\beta$. A fixed penalty and a hard constraint are not identical practical procedures. An adaptive controller can increase $\beta$ when measured KL exceeds a target, but lag, noise, and a poorly chosen step size can cause oscillation.

Value loss trains the critic, and entropy may regularize the actor. Their scales affect optimization even though they play different conceptual roles.

> **Figure placeholder F02 — PPO clipping by advantage sign.** Plot the unclipped and clipped surrogates against probability ratio for one positive and one negative advantage. Mark the flat regions and show that clipping is one-sided for each sign.

**Checkpoint.** Why can KL increase even when many samples are clipped? Parameters are shared, unsampled actions are unconstrained, and clipping removes incentives rather than forbidding parameter movement.

#### 5.4 Derivation: the performance-difference identity

For any trajectory under $\pi'$, the old-policy advantage satisfies

$$
A^\pi(s_t,a_t)
=\mathbb E[r_t+\gamma V^\pi(s_{t+1})-V^\pi(s_t)\mid s_t,a_t].
$$

Multiply by $\gamma^t$, sum, and take expectations. The value terms telescope. With bounded values and $\gamma<1$, the terminal tail vanishes, leaving

$$
\mathbb E_{\pi'}\sum_t\gamma^t A^\pi(s_t,a_t)
=J(\pi')-\mathbb E_{s_0}V^\pi(s_0)
=J(\pi')-J(\pi).
$$

Rewriting the discounted sum using normalized occupancy gives the identity in Section 5.1. Notice that the states are sampled under the new policy, while the advantage evaluates actions relative to the old policy's continuation.

The practical surrogate replaces the new state distribution with the old one. A small policy change is intended to make this approximation reasonable. The trust-region argument therefore concerns occupancy shift as well as individual action ratios.

**Boundary case:** in a one-step contextual bandit, there is no policy-induced future state distribution. This removes one major source of approximation, making finite bandits useful for isolating the remaining effects of clipping and normalization.

### Research-depth mathematical modules — taken just in time

#### Module M1 — Linear algebra and differential calculus for ML research

**Use when:** inspecting logits, LoRA, curvature, or gradient code.

Vectors represent coordinates in a space. An inner product $u^\top v$ measures alignment; the Euclidean norm is $\sqrt{v^\top v}$. Matrix multiplication represents a linear map. A projection onto columns of a full-column-rank matrix $X$ is $X(X^\top X)^{-1}X^\top$; use QR or SVD numerically rather than explicitly inverting a poorly conditioned matrix.

For a symmetric matrix, orthonormal eigenvectors diagonalize its action. The singular value decomposition $W=U\Sigma V^\top$ exists for any real matrix. Keeping the largest $r$ singular values gives the best rank-$r$ approximation in Frobenius norm. LoRA constrains an update to rank at most $r$, but gradient descent on its factors does not automatically produce the truncated SVD of a full-fine-tuning update.

The Jacobian of $f:\mathbb R^n\to\mathbb R^m$ contains $\partial f_i/\partial x_j$. Reverse-mode autodiff efficiently computes vector-Jacobian products. For a scalar loss, its gradient is a vector; its Hessian contains second derivatives. The local approximation is

$$
L(\theta+\Delta)\approx L(\theta)+g^\top\Delta+
\tfrac12\Delta^\top H\Delta.
$$

Ill-conditioning means some directions change the function much faster than others. A quadratic with Hessian eigenvalues 1 and 1000 allows a much smaller stable gradient-descent step along the second direction.

For logits $z$ and softmax probabilities $p_i=e^{z_i}/\sum_j e^{z_j}$,

$$
\frac{\partial\log p_k}{\partial z_j}=\mathbf1[j=k]-p_j.
$$

This identity explains both supervised cross-entropy gradients and the Bernoulli policy-gradient example.

**Worked exercise.** If $p=(0.7,0.2,0.1)$ and the target is class 2, the cross-entropy gradient with respect to logits is $(0.7,-0.8,0.1)$. It sums to zero because adding a constant to all logits changes no probabilities.

#### Module M2 — Probability, statistics, and Monte Carlo estimation

**Use when:** comparing estimators, interpreting learning curves, or evaluating small improvements.

A random variable maps outcomes to numbers. Conditional expectation $\mathbb E[Y\mid X]$ is itself a function of $X$. The laws

$$
\mathbb E[Y]=\mathbb E[\mathbb E[Y\mid X]],
\qquad
\text{Var}(Y)=
\mathbb E[\text{Var}(Y\mid X)]
+\text{Var}(\mathbb E[Y\mid X])
$$

separate within-prompt and across-prompt variation. Sampling more responses per prompt reduces one source of uncertainty while sampling more prompts addresses the other.

Covariance measures joint variation:
$\text{Cov}(X,Y)=\mathbb E[(X-\mathbb EX)(Y-\mathbb EY)]$.
Subtracting a correlated control variate can reduce variance. For $X-c(Z-\mathbb EZ)$, the variance-minimizing scalar coefficient is $\text{Cov}(X,Z)/\text{Var}(Z)$, when the denominator is positive.

An estimator's bias is $\mathbb E\hat\theta-\theta$; its mean squared error is variance plus squared bias. Consistency means convergence to the target as sample size grows, not unbiasedness at every finite size. For independent samples with finite variance, the standard error of their mean is $s/\sqrt n$.

Jensen's inequality says $f(\mathbb EX)\leq\mathbb E f(X)$ for convex $f$. Applying it to $-\log$ helps prove KL nonnegativity. It also warns that averaging a ratio or logarithm differs from taking that function of averages.

For independent observations in $[0,1]$, Hoeffding's inequality gives

$$
\Pr(|\bar X-\mathbb EX|\geq\varepsilon)
\leq2e^{-2n\varepsilon^2}.
$$

This is a conservative distribution-free concentration statement, not a model of all training noise. Shared prompts, duplicated examples, and common seeds can create dependence.

Confidence intervals quantify uncertainty under a sampling model. They do not assign a frequentist posterior probability to the fixed parameter. Bootstrap resampling approximates a sampling distribution; resample at the independent unit, often prompt or task rather than token. For comparisons on the same prompts, bootstrap paired differences.

Hypothesis tests require a null and a test statistic. Repeatedly trying metrics or seeds until one is significant inflates false discoveries. Predefine a primary endpoint, report effect size, and correct or clearly label exploratory comparisons.

**Example.** An accuracy estimate of 0.6 from 100 independent tasks has approximate standard error $\sqrt{0.6(0.4)/100}\approx0.049$. A two-point gain is not compelling on that sample alone. With 10,000 tasks, the analogous standard error is about 0.0049, though dataset biases do not disappear with sample size.

#### Module M3 — Information theory and variational views

**Use when:** deriving DPO, interpreting KL regularization, or comparing divergences.

An $f$-divergence has form $D_f(p\|q)=\mathbb E_q[f(p/q)]$ for convex $f$ with $f(1)=0$. KL uses $f(u)=u\log u$. Different curvature near and far from $u=1$ gives different penalties for likelihood-ratio changes.

Mutual information $I(X;Y)=D_{\mathrm{KL}}(p(x,y)\|p(x)p(y))$ measures statistical dependence. It does not by itself establish causal influence.

The variational identity central to this course is

$$
\beta\log\mathbb E_{y\sim q}e^{R(y)/\beta}
=\max_p\{\mathbb E_p R-\beta D_{\mathrm{KL}}(p\|q)\}.
$$

Let $Z=\sum_yq(y)e^{R(y)/\beta}$ and $p^*(y)=q(y)e^{R(y)/\beta}/Z$. Substitution gives

$$
\mathbb E_pR-\beta D_{\mathrm{KL}}(p\|q)
=\beta\log Z-\beta D_{\mathrm{KL}}(p\|p^*).
$$

Nonnegativity of KL proves the result whenever the partition function is finite and support conditions hold. This is a distribution-space optimum; a neural policy may not represent it or reach it.

Maximum entropy is the special case with a uniform reference on a finite space: KL to uniform equals a constant minus entropy. A nonuniform reference also encourages behaviors already likely under that reference.

**Exercise.** For two responses with equal reference probability, rewards 0 and 1, and $\beta=1$, the preferred response has optimal probability $e/(1+e)\approx0.731$. Increasing $\beta$ moves the solution toward the reference.

#### Module M4 — Optimization and stochastic approximation

**Use when:** comparing PPO/TRPO, adaptive penalties, or stability arguments.

For $\min_\theta f(\theta)$ subject to $g(\theta)\leq0$, the Lagrangian is $f(\theta)+\lambda g(\theta)$, $\lambda\geq0$. KKT conditions include stationarity, feasibility, dual feasibility, and complementary slackness. Under suitable convexity and constraint qualifications these characterize optima; neural training usually lacks those global guarantees.

Natural gradient replaces Euclidean parameter geometry with the local geometry of the distribution. If

$$
F=\mathbb E_{\pi_\theta}
[\nabla\log\pi_\theta\,\nabla\log\pi_\theta^\top],
$$

then locally KL behaves like $\tfrac12\Delta^\top F\Delta$. A constrained linearized improvement yields a direction proportional to $F^{-1}g$, using damping or a suitable generalized inverse when necessary. Forming this matrix is infeasible for large models; practical algorithms approximate its action.

Mirror descent chooses an update by balancing a local linear objective against a Bregman divergence. With KL geometry over a finite probability simplex, reward ascent leads to exponential reweighting $p_{\mathrm{new}}(a)\propto p_{\mathrm{old}}(a)e^{\eta r(a)}$. This connects optimization geometry with the variational optimum.

Stochastic approximation uses noisy gradients. Classic convergence results often assume diminishing step sizes satisfying $\sum_t\eta_t=\infty$ and $\sum_t\eta_t^2<\infty$, along with additional smoothness and noise assumptions. Constant learning rates and nonstationary rewards do not automatically inherit those guarantees.

**Exercise.** Explain why a noisy estimate of KL combined with an aggressive dual update can oscillate: policy changes increase KL, a delayed large penalty overcorrects, then a low measured KL reduces the penalty again.

**Part I mastery check:** derive the Bernoulli policy gradient, prove baseline cancellation, calculate GAE by hand, and explain which distribution appears in every KL and importance ratio.

<a id="part-ii"></a>
## Part II — Language-Model Training Mechanics

**Chapter navigation:** [6 · Transformer training at the token level](#chapter-6) · [7 · Optimization](#chapter-7) · [8 · Parameter-efficient fine-tuning](#chapter-8) · [9 · Training systems fundamentals](#chapter-9)

**Learning goals:** inspect a next-token loss, choose an update parameterization, and account for training memory.

<a id="chapter-6"></a>
### Chapter 6 — Transformer training at the token level

#### 6.1 Tokenization and causal prediction

A tokenizer maps text to token IDs. The same visible string can acquire different IDs when whitespace, special tokens, or chat templates differ. Save the tokenizer and template with the checkpoint.

A transformer turns token and position representations into hidden vectors using attention and feed-forward layers. In causal attention, position $t$ can access positions up to $t$, but not future positions. For queries, keys, and values,

$$
\text{Attention}(Q,K,V)
=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}+M\right)V,
$$

where forbidden entries of $M$ are negative infinity. A padding mask is a separate concept: it excludes positions that are not real content.

With input tokens $(z_0,\ldots,z_{L-1})$, the logits at position $t$ predict $z_{t+1}$. Teacher forcing supplies the observed prefix rather than sampled previous predictions. A shift error can make a model appear to learn rapidly by revealing the target.

#### 6.2 The mask is part of the objective

Let $m_{it}$ be 1 when target token $t$ of sequence $i$ is supervised. A token-average SFT loss is

$$
L_{\mathrm{SFT}}=
-\frac{\sum_{i,t}m_{it}\log\pi_\theta(z_{i,t}\mid z_{i,<t})}
{\sum_{i,t}m_{it}}.
$$

Prompt masking sets user/prompt targets to zero while keeping them in the context. Completion-only loss trains the response. Assistant-only loss extends this to multiple assistant messages in a conversation. These are target masks, not instructions to remove the context from attention.

| Position type | Visible as context? | Usually receives SFT loss? |
|---|---:|---:|
| System instruction | Yes | No |
| User prompt | Yes | No |
| Assistant response | Yes | Yes |
| Tool result | Yes | Usually no |
| Padding | No | No |

If one example has 10 supervised tokens and another 100, token averaging gives the latter ten times as many loss terms. Sequence averaging first normalizes each sequence and therefore defines different weighting.

**Numerical example.** Two target probabilities 0.8 and 0.25 yield loss $-(\log0.8+\log0.25)/2\approx0.805$. Adding 100 padding tokens must not reduce that loss.

> **Figure placeholder F03 — Shifted targets and masks.** Show one system/user/assistant sequence in aligned token, target, attention, and loss-mask rows. Caption: “A token can condition prediction without being a training target.”

#### 6.3 Worked alignment example

Suppose a tokenized training example is $[B,U_1,U_2,A_1,A_2,E]$: beginning marker, two user tokens, two assistant tokens, and EOS. The input is the first five tokens and the target is the last five. A completion-only target mask is $[0,0,1,1,1]$, because the first supervised prediction is $A_1$ from the prefix ending in $U_2$.

The assistant's first token is predicted before it is present in the input prefix. This is where off-by-one errors often occur. Printing decoded input-target pairs is more informative than checking tensor shapes alone.

If the batch pads after EOS, padding targets get mask zero. EOS itself usually remains supervised because learning to stop is part of the response distribution. If every EOS is masked out, the model may learn content while failing to terminate appropriately.

A causal transformer adds residual connections, normalization, positional information, and feed-forward transformations around attention. Those details shape optimization, but none changes the input-target shift. Preserve the alignment test when replacing the tiny model with a more sophisticated architecture.

#### Lab 1 [A; optional B/C] — Tiny autoregressive model

**Objective:** train a causal LM and inspect token cross entropy.

**A setup:** use a character vocabulary for digits, arithmetic symbols, a separator, and EOS. Generate strings such as “2+3=5;”. Start with a bigram table or a small GRU if necessary; then implement a one-block transformer with width 32, two attention heads, and maximum length 32. Framework choice is flexible; the objective is fixed.

**Procedure:**

1. Generate 1,000 arithmetic strings and hold out operand pairs before making variants.
2. Create shifted inputs/targets and print one decoded example with its masks.
3. Overfit eight examples. A declining loss is only a first check.
4. Test causality by changing a future token; earlier logits must be unchanged with dropout disabled.
5. Train on the full training split and report teacher-forced loss and free-running exact-answer accuracy separately.
6. Compare token and sequence averaging on a deliberately mixed-length dataset.

**Submit:** code, configuration, a decoded mask example, loss curve, generated samples, and a short failure analysis. **Checks:** padding invariance, no target leakage, finite gradients, and reproducible evaluation from a saved checkpoint. **B/C extension:** repeat with a small pretrained model and its documented chat template.

<a id="chapter-7"></a>
### Chapter 7 — Optimization

#### 7.1 What an optimizer changes

SGD applies $\theta\leftarrow\theta-\eta g$. Momentum maintains a smoothed direction, reducing oscillation when gradients repeatedly point in similar directions. Adam keeps first and second moments:

$$
m_t=b_1m_{t-1}+(1-b_1)g_t,\quad
v_t=b_2v_{t-1}+(1-b_2)g_t^2,
$$

with bias-corrected estimates $\hat m_t,\hat v_t$, and an update proportional to $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$. AdamW decouples weight decay from this adaptive gradient scaling. Weight decay is not a direct KL constraint on the policy.

Warmup gradually increases the learning rate. A schedule later reduces it or changes it with a defined budget. In online RL, the target distribution also evolves, so a schedule borrowed from fixed-data SFT may behave differently.

#### 7.2 Accumulation, clipping, and precision

Gradient accumulation forms a larger effective batch from microbatches. To match a full-batch token-average loss, weight each microbatch by its number of supervised tokens relative to the total. Dividing every microbatch loss by the number of microbatches is equivalent only when their denominators match.

Global norm clipping applies

$$
g\leftarrow g\min\left(1,\frac{c}{\|g\|_2+\epsilon}\right).
$$

It changes the update, not the loss. Clip after accumulation; with loss scaling, unscale before clipping.

Mixed precision lowers memory and can accelerate hardware. FP16 and BF16 have different exponent and mantissa tradeoffs. Numerical stability remains important: compute log softmax directly rather than taking the log of an underflowed softmax; use log-sum-exp for normalization; inspect extreme likelihood ratios.

**Example.** If microbatch A contains 10 targets and mean loss 1, while B contains 90 targets and mean loss 3, the full token-average loss is 2.8, not 2.

**Checkpoint.** A loss spike follows a longer batch. Inspect supervised-token count, gradient norm, denominator, accumulation scaling, and truncation before concluding that the learning rate is too high.

#### 7.3 Separate gradient scale from objective scale

Multiplying an entire loss by a positive constant preserves its minimizers in an exact unconstrained setting, but it need not preserve a finite training trajectory. SGD's effective step changes unless the learning rate is adjusted. Adam's approximate scale invariance is affected by epsilon, weight decay, clipping, finite precision, and changing gradient statistics.

Similarly, changing from a sum to an average may look like a harmless constant for fixed-size batches. With variable response lengths, the denominator can vary across batches or examples, changing relative weighting and optimizer behavior.

**Worked experiment:** use a two-dimensional quadratic with curvature 1 in one direction and 100 in the other. Run SGD at a stable step, then multiply the loss by 10 without changing the step. The steeper direction can become unstable. Restore the original effective step and compare trajectories.

For language-model RL, log the loss denominator, gradient norm before clipping, fraction of updates clipped, and learning rate. These measurements help distinguish a changed statistical objective from a changed numerical update scale.

<a id="chapter-8"></a>
### Chapter 8 — Parameter-efficient fine-tuning

#### 8.1 LoRA as a parameterization

For a frozen weight $W_0\in\mathbb R^{d_{\mathrm{out}}\times d_{\mathrm{in}}}$, LoRA learns

$$
W=W_0+\frac{\alpha}{r}BA,
\quad
B\in\mathbb R^{d_{\mathrm{out}}\times r},
\quad
A\in\mathbb R^{r\times d_{\mathrm{in}}}.
$$

The trainable parameter count is $r(d_{\mathrm{in}}+d_{\mathrm{out}})$. For a 1024-by-1024 layer and rank 8, this is 16,384 parameters rather than 1,048,576. Usually one factor is initialized to zero and the other randomly, so the initial update is zero while gradients can begin moving one factor. Initializing both factors to zero creates a dead start.

Rank limits the update subspace. Alpha changes update scaling. Placement determines which transformations can adapt: attention projections and feed-forward layers affect different computations. These are experimental variables, not interchangeable implementation details. See [LoRA][lora].

#### 8.2 Quantization and merging

Quantization stores weights using fewer bits plus scale and metadata. Four-bit storage does not imply all arithmetic, gradients, or activations use four bits. QLoRA combines quantized frozen base weights with trainable low-rank adapters, with specific quantization and memory techniques in its original design. [QLoRA][qlora]

Eight-bit and four-bit methods differ in representation and support. Quantization error can change policy probabilities even when top responses remain the same, which matters for RL importance ratios.

Adapter merging computes $W_0+\alpha BA/r$ when the representation permits it. Merging into a quantized model may require dequantization and requantization, introducing differences. Validate logits before and after merging.

Full tuning can access directions excluded by a low-rank parameterization, but costs more optimizer state. Choose based on a measured task/compute tradeoff.

#### 8.3 Derive the adapter gradients

Write a linear layer as $y=(W_0+cBA)x$, where $c=\alpha/r$, and let $g_y=\partial L/\partial y$. For a single example,

$$
\frac{\partial L}{\partial B}=c\,g_y(Ax)^\top,
\qquad
\frac{\partial L}{\partial A}=c\,B^\top g_y x^\top.
$$

These expressions explain the initialization behavior directly. With $B=0$ and nonzero $A$, $B$ receives a gradient, while $A$'s initial gradient is zero. After $B$ moves, both factors can learn.

The factorization is nonunique: multiplying $B$ by a nonzero scalar and dividing $A$ by it leaves $BA$ unchanged. Optimizer state and weight decay on the factors can nevertheless change the training path. A low-rank parameterization has optimization geometry beyond the rank constraint alone.

**Research exercise:** create two factorizations with identical $BA$ but different factor scales. Apply one SGD or Adam-style update to each and compare the resulting effective weight changes. State which optimizer assumptions would be needed for equivalence.

#### Lab 3 [A; optional B/C] — LoRA from scratch

**Objective:** implement and inspect low-rank update matrices.

1. Fit a small linear map using full tuning.
2. Freeze a different starting map and learn factors $A,B$ for ranks 1, 2, and 4.
3. Compare reconstruction error, parameter count, and singular values of $BA$.
4. Initialize both factors to zero once and explain the failed gradients.
5. Merge the learned update; compare predictions before and after merging.
6. Add adapters to the tiny LM from Lab 1, freezing and reporting all other parameters.

**Submit:** rank/error table, gradient checks, trainable-parameter list, and merge agreement. **B/C extension:** measure actual memory for adapter and full tuning; include activations and optimizer state, not just saved checkpoint size.

<a id="chapter-9"></a>
### Chapter 9 — Training systems fundamentals

#### 9.1 Memory accounting

Training memory consists of parameters, gradients, optimizer states, activations, temporary workspaces, and runtime overhead. Under one common mixed-precision accounting, each parameter needs 2 bytes of model weight, 2 of gradient, 4 of master weight, and 8 of Adam moments, totaling about 16 bytes before activations. Other implementations differ.

At one billion parameters, that estimate is about 16 GB in decimal units before activation memory. Inference with a frozen quantized model has a different footprint.

Activation checkpointing discards selected intermediate activations and recomputes them during backpropagation. It exchanges compute for memory. Long contexts can make attention and saved activations dominate even with adapters.

#### 9.2 Ways to split work

| Strategy | What is split | Main cost |
|---|---|---|
| Data parallelism | Examples across replicated models | Gradient synchronization |
| Tensor parallelism | Matrix operations within layers | Frequent communication |
| Pipeline parallelism | Layers across workers | Bubbles and activation transfers |
| Sequence/context parallelism | Sequence-related work | Attention and activation communication |
| Optimizer/state sharding | Parameters, gradients, or optimizer states | Gathering and redistribution |

ZeRO stages progressively shard optimizer states, gradients, and parameters. FSDP is a fully sharded data-parallel approach with parameter gathering and resharding behavior determined by configuration. Names alone do not determine peak memory; the communication schedule matters. [ZeRO][zero]

**Exercise.** Sketch memory for a frozen base plus adapters and for full tuning. Which terms shrink with adapters? Trainable gradients and optimizer state shrink; base storage and much activation memory remain.

**Part II deliverable:** a tiny model whose data alignment, gradients, checkpoint reload, and memory requirements you can explain.

#### 9.3 A memory budget is a sum of peaks

Suppose frozen base weights occupy 2 GB, trainable adapter states occupy 0.1 GB, and peak activations occupy 3 GB. Reducing adapter state by half saves only 0.05 GB. Reducing activation storage by half saves 1.5 GB. Parameter-efficient tuning does not guarantee that trainable parameters are the dominant term.

Peak memory depends on when allocations coexist. A parameter all-gather can overlap with saved activations; an inference KV cache can remain resident when training begins. Adding isolated component sizes measured at different times can underestimate the real peak.

Use a budget table with component, bytes per element, element count, precision, lifetime, and whether it is sharded. Then run a small measured pilot. Distinguish decimal GB from binary GiB when comparing estimates with runtime reports.

**Exercise:** estimate memory for one full-tuning step and one adapter-tuning step at two context lengths. Explain which quantities scale with parameter count, batch size, and sequence length. The answer should predict a trend before you inspect a profiler.

<a id="part-iii"></a>
## Part III — Supervised Post-Training

**Chapter navigation:** [10 · Instruction tuning and SFT](#chapter-10) · [11 · Data quality](#chapter-11) · [12 · Behavioral cloning](#chapter-12) · [13 · Distillation](#chapter-13)

**Learning goals:** design a demonstration distribution, distinguish imitation from recovery, and transfer useful behavior from a teacher.

<a id="chapter-10"></a>
### Chapter 10 — Instruction tuning and SFT

#### 10.1 Demonstrations define behavior

An instruction example has a context and a desired continuation. The training objective says “increase the likelihood of this continuation,” not “understand why this answer is good.” If demonstrations systematically include verbosity, unjustified confidence, or unnecessary tool calls, those become statistical targets too.

A chat template serializes roles and boundaries. Use the same role syntax at training and inference. Multi-turn examples should preserve which information was available before each assistant message. A tool result must follow the call that caused it.

Packing concatenates short examples to use sequence space efficiently. Decide whether examples may attend across boundaries. Independent examples often need block-diagonal attention or an equivalent separation scheme. Merely inserting EOS does not mathematically prohibit cross-example attention.

#### 10.2 Context, optimization, and retention

Context truncation can remove the instruction, truncate the answer, or hide a tool result. Each option changes the learning problem. Count how many examples and target tokens are affected.

More epochs repeatedly expose the same examples. They can improve fit or increase memorization and forgetting. Learning rate, number of updates, and data diversity interact: “three epochs” is not a portable prescription across dataset sizes.

Catastrophic forgetting means performance on previously supported behaviors deteriorates after adapting to new data. Measure retention on a fixed set; consider mixed-domain data, smaller updates, or a regularizer. Do not infer retention from the training loss.

**Worked example.** If a conversation has user text of length 20 and assistant text of length 10, assistant-only training supervises 10 tokens. Accidentally training all 30 changes the relative weight of instruction copying and answering.

#### 10.3 SFT as a conditional distribution projection

For a fixed prompt $x$, imagine demonstrations contain response A 80% of the time and response B 20% of the time. An unrestricted maximum-likelihood policy recovers that distribution. It does not infer that A is intrinsically better; it matches observed frequencies.

If B is a mistake, filtering or weighting changes the target distribution. If B is a valid alternative, removing it narrows diversity. A chosen-only preference dataset used as SFT demonstrations already includes a selection mechanism.

This viewpoint clarifies why low SFT loss can coexist with poor task success. The model can faithfully imitate a flawed or mismatched demonstration distribution. Evaluation must assess the intended task, not just agreement with training text.

**Practice:** create two demonstrations for the same arithmetic prompt, one correct and one wrong, with controlled frequencies. Fit a finite policy by maximum likelihood and verify the predicted distribution. Then compare with a correctness reward. The difference comes from the supervision target, not from the optimizer's name.

#### Lab 2 [A; optional B/C] — Tiny instruction SFT

**Objective:** format examples and tune a small instruction model.

1. Extend Lab 1 with two tasks, such as “add” and “reverse digits.”
2. Create explicit task and response delimiters; construct response-only masks.
3. Hold out operand pairs and string patterns, not just randomly duplicated rows.
4. Compare the pretrained tiny LM, full-response SFT, and an intentionally wrong mask.
5. Evaluate formatting validity and semantic correctness separately.
6. Add a retention set from the original LM task and measure forgetting.

**Submit:** data schema, decoded templates, mask visualizations, pre/post scores, and representative mistakes. **Checks:** no labels outside assistant targets, no evaluation duplicates, and correct EOS training. **B/C extension:** adapt a small pretrained model with LoRA and compare with its frozen baseline.

<a id="chapter-11"></a>
### Chapter 11 — Data quality

#### 11.1 Quality is conditional on the target

A high-quality explanation can still be poor training data for a terse extraction task. Evaluate correctness, relevance, difficulty, coverage, and style against the intended deployment distribution.

Deduplication prevents a repeated example from receiving accidental extra weight. Exact hashing catches identical rows; normalized or semantic similarity catches templates and paraphrases. Deduplicate related examples before splitting whenever possible. Contamination includes test answers and near-equivalent tasks, not just exact strings.

Domain balance controls what the average loss prioritizes. A 90% coding mixture spends most updates on coding even if the desired deployment distribution is balanced. You can deliberately oversample a difficult domain, but report both the sampling mixture and its motivation.

#### 11.2 Difficulty, curricula, and synthetic data

A curriculum changes the ordering or sampling probabilities of examples. Easy-to-hard is one hypothesis; interleaving can prevent forgetting. Define difficulty through measured model behavior or task structure, not merely response length.

Synthetic data can expand coverage cheaply, but shares the teacher's errors and stylistic biases. Filtering can improve average quality while removing useful hard examples. A teacher that rejects unfamiliar but correct solutions narrows the learned distribution.

**Worked example.** A filter accepts 95% of easy tasks and 20% of difficult tasks. Even if the original pool is balanced, the accepted pool is approximately 83% easy. A claimed “better dataset” may primarily be a changed curriculum.

**Exercise.** Build a small data card: source, task distribution, label process, deduplication, contamination checks, acceptance rates by domain, and known omissions. Predict which deployment failures each omission could produce.

<a id="chapter-12"></a>
### Chapter 12 — Behavioral cloning

#### 12.1 Demonstrations as trajectories

Behavior cloning minimizes

$$
L_{\mathrm{BC}}(\theta)=
-\mathbb E_{(s,a)\sim d_{\mathrm{expert}}}
\log\pi_\theta(a\mid s).
$$

The expectation is under expert-visited states. At deployment, actions come from the learner, whose errors can move it into unfamiliar states. This covariate shift compounds over a long horizon.

If the learner has an independent probability $1-\varepsilon$ of making the right decision at each of $H$ steps, uninterrupted success is $(1-\varepsilon)^H$. At $\varepsilon=0.02,H=50$, it is about 0.364. Real errors are dependent, but the example explains why local accuracy is insufficient.

#### 12.2 Recovery and dataset aggregation

DAgger alternates learner rollouts with expert labeling of the states the learner actually encounters, aggregating those states into training data. The expert supplies corrective actions where the learner needs them. Its theoretical assumptions and supervision cost matter. [DAgger][dagger]

Teacher trajectories can include successful paths, failed paths, and recovery. A failed trace is useful if it teaches an appropriate correction or supplies a negative example. Blind SFT on every failed action reinforces failure.

Agent trajectory SFT supervises model-selected actions and language. Observations remain context. If a trajectory is compressed, ensure the learner sees sufficient information to infer why the demonstrated action was appropriate.

**Checkpoint.** A cloned agent succeeds when initialized halfway through an expert trajectory but fails from the beginning. Investigate state-distribution shift and recovery behavior, not just action syntax.

#### 12.3 A concrete DAgger round

Start with a policy cloned from expert traces. Run it in the environment and save every state it visits, including mistaken states. Ask the expert for the appropriate action at those states. Add the new state-action pairs to the aggregated dataset and retrain.

Suppose the learner looks up the wrong record. The expert action from the original demonstration was “calculate,” but from the learner's current state the right action may be “correct the lookup.” Copying the original action without conditioning on the new state would produce an invalid label.

The expert must be competent in the visited states, and labeling may be expensive. Some learner states are unrecoverable under the remaining budget. The correct demonstration may then be an honest failure report or restart, depending on the task.

**Exercise:** compare adding ten more expert-success traces with adding ten expert corrections at common learner mistakes. Evaluate both on learner-induced states. This directly tests the coverage argument behind dataset aggregation.

<a id="chapter-13"></a>
### Chapter 13 — Distillation

#### 13.1 Hard targets, soft targets, and sequences

Hard-label distillation trains on teacher outputs. Soft-target distillation matches a teacher distribution over tokens. With temperature $\tau$,

$$
L_{\mathrm{KD}}=\tau^2
D_{\mathrm{KL}}(p_T^\tau(\cdot\mid h)\|p_S^\tau(\cdot\mid h)).
$$

The conventional $\tau^2$ factor compensates for gradient scaling in the softened-logit setting; temperature also changes which distinctions are emphasized. Exact logit matching requires access to compatible token distributions.

Sequence-level distillation generates complete responses and trains on them. It works across some teacher/student tokenizer differences because the interface is text, though tokenization still changes the student's loss weighting.

#### 13.2 Teacher sampling and student coverage

Greedy teacher sampling gives one mode. Diverse sampling exposes alternatives but can introduce lower-quality examples. Reasoning traces may transfer useful intermediate structure, yet a fluent trace can be invalid. Verify outcomes and, where feasible, important intermediate steps.

On-policy distillation obtains teacher supervision at states or prefixes reached by the student. This addresses student-teacher mismatch: a teacher's ideal trajectory may never visit the student's common mistakes. For an agent, ask the teacher how to recover from a real failed tool call.

**Worked example.** A teacher always solves a task in five steps; a smaller student sometimes corrupts step two. Training only on flawless traces never teaches what to do from the corrupted state. Collecting teacher corrections at those student states changes the coverage directly.

**Exercise.** Compare three datasets at equal supervised-token count: greedy teacher responses, diverse verified responses, and teacher corrections to student rollouts. Specify an evaluation that distinguishes style imitation from task improvement.

**Part III mastery check:** explain which state distribution every supervised loss uses and how that distribution differs from deployment.

<a id="part-iv"></a>
## Part IV — Human Preferences and Reward Modeling

**Chapter navigation:** [14 · Preference data](#chapter-14) · [15 · Reward models](#chapter-15) · [16 · Reward-model pathology](#chapter-16) · [17 · Process reward models](#chapter-17)

**Learning goals:** convert feedback into a statistical model, identify reward-model uncertainty, and distinguish outcome supervision from process supervision.

<a id="chapter-14"></a>
### Chapter 14 — Preference data

#### 14.1 Feedback formats encode different information

Pairs ask which of two responses is better. Rankings order several responses. Scalar ratings require a calibrated scale. Accept/reject labels assess individual responses against a threshold. These formats are not freely interchangeable: a response can be preferred in a pair while both options are unacceptable.

Best-of-N selection produces a winner from a candidate set, but a winner is not necessarily a calibrated high-quality example. Record the losing candidates and the sampling procedure if you want to analyze the selection.

Human annotators and AI judges both require a rubric. Split correctness, completeness, style, and compliance where those dimensions can conflict. Record ties and ambiguity instead of forcing arbitrary binary labels.

#### 14.2 Noise and disagreement

Random mistakes are label noise. Persistent disagreement may reflect different legitimate preferences or inconsistent instructions. A single scalar reward assumes that the chosen representation can summarize those preferences adequately.

Position randomization helps expose order effects. Repeated labels estimate consistency. Agreement metrics summarize reproducibility, not truth: two judges can agree on the same misconception.

**Example.** One response is correct and terse; another is partly wrong but more detailed. If annotators reward detail without a correctness rule, a learned reward may optimize polish at the expense of truth.

#### Lab 4 [A] — Preference dataset

**Objective:** create chosen/rejected pairs and measure label noise.

1. Generate 100 prompts with two to four candidate answers from a tiny task.
2. Define an executable “true utility” using correctness and an explicit optional cost.
3. Create pair labels, then flip 0%, 10%, and 30% with a fixed random seed.
4. Add a systematic bias condition that prefers longer responses.
5. Record ties, candidate order, source policy, rubric, and noise condition.
6. Hold out whole prompts for later reward-model evaluation.

**Submit:** dataset, annotation schema, agreement/noise summary, and examples where preference differs from absolute acceptability. **Check:** chosen/rejected labels reverse consistently when presentation order is swapped.

<a id="chapter-15"></a>
### Chapter 15 — Reward models

#### 15.1 Bradley–Terry from score differences

Let $r_\phi(x,y)$ be a scalar score. Bradley–Terry models pair preference as

$$
\Pr(y_w\succ y_l\mid x)=
\frac{e^{r_\phi(x,y_w)}}{e^{r_\phi(x,y_w)}+e^{r_\phi(x,y_l)}}
=\sigma(r_w-r_l).
$$

For observed winners, maximum likelihood gives

$$
L_{\mathrm{RM}}=-\log\sigma(r_w-r_l).
$$

If the score difference is 2, the modeled preference probability is about 0.881. The loss is about 0.127. The derivative with respect to the difference is $\sigma(2)-1\approx-0.119$, encouraging a larger winning margin.

Adding the same prompt-dependent constant to all response rewards changes no pair probabilities. Pairwise learning does not identify absolute reward offsets. Score scale also depends on noise modeling and regularization.

#### 15.2 Architecture and calibration

A sequence-classification reward model typically encodes prompt and response and applies a scalar head to a chosen pooled representation or terminal hidden state. The pooling and truncation choices affect what it can score. A lost final answer can make a seemingly adequate context window unusable.

Normalization can standardize scores, but changing reward scale changes the policy tradeoff with KL. Calibration asks whether predicted pair probabilities correspond to empirical preference frequencies. Pair accuracy alone says nothing about calibration.

Generalization should be tested on new prompts, response styles, and policies. A reward model trained on weak candidates may extrapolate poorly to optimized outputs. Ensembles can expose disagreement, but shared data and architectures can produce correlated mistakes.

#### 15.3 Calibrated preference is not calibrated utility

Suppose a reward difference predicts that A beats B with probability 0.8. This is a statement about the modeled comparison process, not that A has an 80% chance of being correct. Both answers could be wrong, with A merely less wrong or more readable.

If your downstream objective is correctness, evaluate the relation between reward scores and executable outcomes separately. A reward model can have excellent pairwise calibration under one rubric and poor correlation with another target.

To test prompt-generalization, hold out prompts. To test policy-generalization, score responses from a new generator. To test feature robustness, create counterfactual pairs where style, length, or position changes independently of correctness.

**Worked diagnostic:** build four response groups crossing correct/incorrect with terse/verbose. If the model ranks verbose incorrect answers above terse correct answers, aggregate pair accuracy may hide the problem when training data rarely contains that crossing.

#### Lab 5 [A; optional B/C] — Reward model

**Objective:** implement and train Bradley–Terry pairwise loss.

1. Start with linear features: correctness indicator, length, formatting, and a deliberately spurious style feature.
2. Fit pairwise logistic loss on Lab 4 data using stable log-sigmoid or log-add-exp.
3. Evaluate pair accuracy and calibration on held-out prompts.
4. Remove the direct correctness feature and observe reliance on proxies.
5. Reverse the style correlation on a test set.
6. Plot reward against known true utility and inspect high-reward errors.

**Submit:** loss derivation, gradient check, calibration bins, and shift analysis. **B/C extension:** replace features with a small neural sequence encoder.

<a id="chapter-16"></a>
### Chapter 16 — Reward-model pathology

#### 16.1 Optimization finds exceptional mistakes

A predictor can be accurate on ordinary samples yet fail on examples selected to maximize it. As the policy improves predicted reward, it can move into regions with larger reward-model errors. This is overoptimization.

Write learned reward as $\hat R=R^*+e$. Selecting high $\hat R$ can select high positive error $e$, especially when candidate search is large. Goodhart's law describes the broader phenomenon that optimizing a proxy can weaken its relationship to the intended quantity.

Length, style, position, and sycophancy biases are concrete examples. A model may prefer a longer answer because training winners were longer, prefer the first answer because of judge ordering, or reward agreement with the user even when the user is wrong.

#### 16.2 Diagnose the mechanism

Track proxy reward against an independent metric. Create counterfactual response pairs where only the suspected feature changes. Separate distribution shift from outright implementation errors.

Ensembles reduce some variance and reveal disagreement. They do not guarantee correctness on inputs outside all members' experience. An uncertainty penalty can help if uncertainty is informative, but a confidently shared blind spot remains dangerous.

**Worked example.** Suppose candidate true qualities are all similar, while reward errors vary. Increasing best-of-N may mostly improve the maximum error. An observed rising reward curve with flat verified accuracy is consistent with this mechanism.

**Exercise.** Construct a table with true correctness, length, learned reward, and judge score. Predict what happens when training optimizes only learned reward. Propose a counterfactual test that isolates length bias.

<a id="chapter-17"></a>
### Chapter 17 — Process reward models

#### 17.1 What a step label means

Outcome supervision labels a completed answer. Process supervision labels intermediate steps. A process reward model can predict local validity, contribution to eventual success, or a human judgment of reasoning quality. These are distinct targets.

Step segmentation is part of the definition. A paragraph, equation, tool call, or token span can be a step. Changing segmentation changes label density and aggregation.

Suppose a trace solves $2x+3=9$: subtract 3, obtain $2x=6$, divide by 2, obtain $x=3$. A locally valid transformation can be checked symbolically. An elegant but unnecessary step may be valid without increasing eventual success.

#### 17.2 Monte Carlo labels and aggregation

One way to label a prefix is to sample continuations and estimate their success rate:

$$
\hat V(h)=\frac1K\sum_{k=1}^K
\mathbf1[\text{continuation }k\text{ succeeds}].
$$

This estimates success under the continuation policy and budget, not universal mathematical correctness. A correct prefix can have low value if the continuation policy is weak.

Learned step probabilities can be aggregated by minimum, product, sum of log scores, or a learned combiner. Each imposes a different preference over trace length and error structure. Multiplying probabilities assumes more structure than merely having individual step classifiers; calibration does not automatically survive aggregation.

Hybrid rewards combine outcomes and process signals. They can improve credit assignment or incentivize a style of “good-looking” steps. Independent outcome checks remain valuable. [Let's Verify Step by Step][prm]

#### 17.3 Process supervision and causal credit are different

Consider a valid intermediate step that is unnecessary. A process-validity classifier should mark it valid. A cost-aware policy might still prefer to omit it. Conversely, a locally invalid step can be followed by a correction that produces a correct final answer.

If every valid step receives positive reward, the agent may insert redundant steps. If trace reward is the product of step-validity probabilities, longer traces can be penalized even when every step is equally reliable. If reward is the minimum, one weakly scored step dominates.

Use a controlled dataset containing short correct traces, long correct traces, corrected mistakes, and fluent invalid traces. Evaluate local validity and final success independently before selecting an aggregation rule.

**Research question:** does a process reward improve the probability of discovering a correct path, or mainly select already correct paths more efficiently? Compare training effects with an inference-only reranking baseline using the same process model.

#### Lab 17 [A; optional B/C] — Process reward model

**Objective:** learn step-level judgments on short reasoning traces.

1. Generate arithmetic transformation chains with known valid steps.
2. Corrupt exactly one step in some chains and track downstream dependence.
3. Train a tiny classifier on state/operation/next-state features.
4. Compare minimum, product, and average aggregation at different trace lengths.
5. Separately estimate prefix success using a stochastic continuation policy.
6. Explain where local validity and continuation success disagree.

**Submit:** labeling rules, step confusion matrix, trace-level performance, and length analysis. **Check:** split by problem template to avoid memorizing exact equations. **B/C extension:** train a small textual step classifier.

**Part IV mastery check:** explain what the reward model can identify, what its scores mean, and how optimization can invalidate its original test accuracy.

<a id="part-v"></a>
## Part V — Classical RLHF

**Chapter navigation:** [18 · The original RLHF pipeline](#chapter-18) · [19 · PPO for language models](#chapter-19) · [20 · Why PPO is difficult](#chapter-20)

**Learning goals:** connect reward learning with policy optimization and implement the tensor flow of PPO.

<a id="chapter-18"></a>
### Chapter 18 — The original RLHF pipeline

#### 18.1 The four roles

A common pipeline starts with a base model, performs SFT, collects preferences, trains a reward model, and optimizes the policy against that reward while limiting departure from an SFT reference. This is one influential construction, not the definition of every feedback-training system. [InstructGPT][instructgpt]

| Role | Trained during policy optimization? | Purpose |
|---|---:|---|
| Actor | Yes | Generate responses |
| Reference | Usually no | Define behavior regularization |
| Reward model | Usually frozen in a phase | Estimate response utility |
| Critic | Yes in actor-critic methods | Estimate expected return |

The old actor is a snapshot for sampling and ratios. It is logically distinct from the fixed reference, even if implementations avoid storing a separate full copy by recording old log probabilities.

#### 18.2 KL as a reward term

For response $y$, define sampled log ratio

$$
k(x,y)=\log\pi_\theta(y\mid x)-\log\pi_{\mathrm{ref}}(y\mid x).
$$

Its expectation under the current policy is the policy-to-reference KL. Individual values can be negative even though the expectation is nonnegative.

The KL-regularized objective can be written
$J_\beta=\mathbb E_\pi[R-\beta k]$.
Since $k$ depends on $\theta$, differentiate carefully:

$$
\nabla J_\beta=
\mathbb E_\pi[(R-\beta k)\nabla\log\pi]
-\beta\mathbb E_\pi[\nabla\log\pi].
$$

The last expectation is zero. Therefore a fresh on-policy score-function estimator may use the detached augmented reward $R-\beta k$. Differentiating only the sampled log ratio while treating the sampled action as fixed is not the full gradient of the distribution-level KL.

Token-level decomposition uses
$k=\sum_t(\log\pi(y_t\mid h_t)-\log\pi_{\mathrm{ref}}(y_t\mid h_t))$.
Assigning each log-ratio penalty at its token permits reward-to-go estimates. With stale rollouts and multiple epochs, this becomes a surrogate around the behavior policy.

#### 18.3 Reward scale and regularization

If all rewards are multiplied by $c>0$, preserving the same distribution-space optimum requires multiplying $\beta$ by $c$. A numerical beta value has no portable meaning without reward scale and aggregation conventions.

#### Lab 7 [A; optional B/C] — KL-regularized RLHF

**Objective:** implement the penalty and inspect its effect.

1. Use a four-answer categorical policy and a frozen reference.
2. Compute exact expected reward and exact KL by enumeration.
3. Calculate the analytic optimum $\pi^*(y)\propto\pi_{\mathrm{ref}}(y)e^{R(y)/\beta}$.
4. Optimize directly, then optimize with sampled score-function gradients.
5. Sweep beta and compare reward, KL, and entropy.
6. Multiply rewards and beta by the same factor and check the predicted optimum.

**Submit:** exact versus learned distributions and Monte Carlo gradient error. **Check:** reward-zero training approaches the reference under the exact objective. **B/C extension:** repeat using the reward model from Lab 5 and audit true utility separately.

<a id="chapter-19"></a>
### Chapter 19 — PPO for language models

#### 19.1 Prepare the rollout once

Sample responses with $\pi_{\mathrm{old}}$. Save response tokens, response mask, old log probabilities, old values, reference log probabilities, and terminal rewards. Disable stochastic layers or otherwise ensure probability calculations match the policy definition.

For token $t$, a common augmented reward is

$$
\tilde r_t=-\beta
\left(\log\pi_{\mathrm{old}}(y_t\mid h_t)
-\log\pi_{\mathrm{ref}}(y_t\mid h_t)\right)
+\mathbf1[t=T]R(x,y).
$$

Use these rewards and old values to compute GAE and value targets. Detach targets before the actor/critic update. If you normalize advantages, document the population: response tokens, sequences, prompts, or the whole batch.

#### 19.2 Actor and critic objectives

At each PPO minibatch update, recompute current log probabilities on the stored responses:

$$
\rho_t=\exp(\log\pi_\theta(y_t\mid h_t)
-\log\pi_{\mathrm{old}}(y_t\mid h_t)).
$$

The actor minimizes minus the clipped surrogate. A simple critic loss is

$$
L_V=\frac{1}{2}\text{maskedmean}
\left[(V_\phi(h_t)-\hat G_t)^2\right].
$$

Some implementations clip value updates around old values. This is a separate design choice from actor clipping. Add an entropy term only with an explicit coefficient and convention. Do not accidentally add a second KL penalty if it is already included in rewards.

| Tensor | Typical response-aligned shape | Gradient? |
|---|---|---|
| Current token log probabilities | batch × response length | Yes |
| Old/reference log probabilities | batch × response length | No |
| Response mask | batch × response length | No |
| Advantages and value targets | batch × response length | No |
| Current values | batch × response length | Yes for critic |
| Terminal reward | batch | No |

#### 19.3 A complete update in pseudocode

```text
freeze behavior snapshot
sample prompts and responses; record behavior probabilities
score responses and compute token KL penalties
compute detached returns and advantages using old values
for a small number of optimization epochs:
    for minibatch in stored rollouts:
        recompute current log probabilities and values
        calculate current/behavior ratios
        calculate masked clipped actor loss
        calculate masked value loss and optional entropy
        update parameters
        measure KL, clip fraction, gradient norm, and value error
discard or explicitly account for stale rollouts before recollecting
```

The initial ratio should be approximately 1 before updating. If not, investigate model versions, sampling transforms, precision, masks, or tokenization.

#### 19.4 A two-step numerical example

Take terminal reward 1, token KL penalties 0.1 and 0.2, $\gamma=1$, and old values $0.4,0.5$. Augmented rewards are $-0.1,0.8$. TD residuals are $0,0.3$. With $\lambda=1$, advantages are $0.3,0.3$, and value targets are $0.7,0.8$.

If the first token ratio becomes 1.3 with $\epsilon=0.2$, its positive-advantage contribution is clipped from 0.39 to 0.36. This is a surrogate calculation, not proof that the new policy's expected reward improved.

**Research question.** How does averaging over tokens versus responses affect the influence of long responses when terminal rewards are identical?

#### 19.5 Why multiple PPO epochs require the old policy

At the beginning of the first update, $\pi_\theta=\pi_{\mathrm{old}}$, so every ratio is 1 and the clipped surrogate has the same local policy-gradient direction as its unclipped form, away from nonsmooth boundaries. After one update, the stored actions are no longer sampled from the current policy.

The ratio retains a connection to the behavior distribution while the clipping rule limits some incentives for large changes. Replacing the denominator with newly recomputed current probabilities would reset every ratio to 1 and remove that mechanism.

Old advantages should also remain fixed within the specified update cycle unless the algorithm explicitly recomputes them. Updating a critic and silently regenerating actor targets halfway through an epoch changes the objective being optimized.

**Debugging exercise:** keep a fixed two-action batch and perform three updates. Print old log probability, current log probability, ratio, advantage, unclipped contribution, clipped contribution, and active gradient branch after each. This is a more reliable explanation of PPO behavior than a single scalar training loss.

<a id="chapter-20"></a>
### Chapter 20 — Why PPO is difficult

#### 20.1 Coupled failure modes

PPO combines several estimators and models. A critic trained on a changing policy can lag. A reward-scale change affects advantages, critic targets, clipping behavior, and the effective KL tradeoff. More PPO epochs improve reuse but increase mismatch with the behavior distribution.

| Symptom | Plausible causes | First measurement |
|---|---|---|
| Ratios differ from 1 before updates | Weight or decoding mismatch | Per-token log-probability difference |
| Reward rises while quality falls | Reward exploitation | Independent task accuracy |
| KL spikes | Large step, stale data, scale change | KL distribution and policy age |
| Critic loss grows | Incorrect targets or nonstationarity | Returns and value calibration |
| Entropy collapses | Narrow rewards or aggressive optimization | Entropy by prompt/difficulty |
| Apparent improvement comes from longer outputs | Length weighting or verifier bias | Reward conditioned on length |

The actor, reference, reward model, and critic can each consume model memory. Shared backbones, offloading, or smaller critics change that footprint, with corresponding engineering tradeoffs.

Generation/training mismatch can arise from sampling temperature, truncated distributions, dropout, mixed precision, or different kernels. Tokenization drift can create an even more fundamental mismatch: probabilities are attached to different events.

#### 20.2 A debugging order

First verify a finite policy with exact expected reward. Then test one update with ratio 1, known rewards, and a fixed batch. Inspect masks, returns, and gradient signs. Add the critic only after the actor path is correct. Add large-scale execution after the small implementation is interpretable.

Advantage normalization subtracts a batch mean and divides by a batch standard deviation. Because these depend on sampled actions, finite-batch properties differ from using an independent state baseline. The practical benefit can still be real; specify what estimator is actually used.

#### Lab 8 [A; optional B/C] — PPO

**Objective:** implement a small policy and inspect every important tensor.

1. Create a two-step environment with four possible action sequences and known terminal rewards.
2. Enumerate its exact expected return and gradient.
3. Add a state-value table, terminal masks, GAE, and clipped actor loss.
4. Verify the Chapter 19 numerical example and both signs of clipping.
5. Compare one versus several PPO epochs on the same rollout batch.
6. Deliberately corrupt old log probabilities and trace the resulting failure.
7. Port the implementation to the tiny LM.

**Submit:** tensor-shape table, one hand-checked update, return/KL/entropy curves, and a failure report. **Checks:** no gradient through old probabilities or advantages, correct termination, and no reward leakage from evaluation.

<a id="part-vi"></a>
## Part VI — Preference Optimization Without Classical RL

**Chapter navigation:** [21 · DPO from first principles](#chapter-21) · [22 · The preference-optimization family](#chapter-22) · [23 · When preference optimization works](#chapter-23)

**Learning goals:** derive DPO, compare preference objectives by their statistical targets, and choose experiments that reveal coverage limits.

<a id="chapter-21"></a>
### Chapter 21 — DPO from first principles

#### 21.1 Solve the KL-regularized problem

For one fixed prompt, consider an unrestricted response distribution $p$:

$$
\max_p\sum_y p(y)R(y)-\beta\sum_y p(y)\log\frac{p(y)}{q(y)}
\quad\text{subject to }\sum_y p(y)=1.
$$

Assume $\beta>0$, positive reference support on responses of interest, and finite normalization. Introduce multiplier $\lambda$. Stationarity gives

$$
R(y)-\beta\left(\log\frac{p(y)}{q(y)}+1\right)+\lambda=0.
$$

After normalizing,

$$
p^*(y)=\frac{q(y)e^{R(y)/\beta}}{Z},\qquad
R(y)=\beta\log\frac{p^*(y)}{q(y)}+\beta\log Z.
$$

The prompt-dependent partition term disappears in reward differences. Combining this reward reparameterization with Bradley–Terry preferences yields DPO. [Direct Preference Optimization][dpo]

#### 21.2 Substitute a trainable policy

Define the response log-ratio score

$$
s_\theta(x,y)=\log\pi_\theta(y\mid x)
-\log\pi_{\mathrm{ref}}(y\mid x).
$$

Then the DPO loss for a chosen/rejected pair is

$$
L_{\mathrm{DPO}}
=-\log\sigma\left(\beta[s_\theta(x,y_w)-s_\theta(x,y_l)]\right).
$$

Reference log probabilities are fixed. Standard sequence-level DPO sums completion-token log probabilities; averaging them changes the objective. Mask prompt and padding tokens consistently.

#### 21.3 The gradient and a numerical example

Let $\Delta=s_w-s_l$ and $z=\beta\Delta$. Then

$$
\nabla_\theta L_{\mathrm{DPO}}
=-\beta\sigma(-z)
\left[\nabla_\theta\log\pi_\theta(y_w\mid x)
-\nabla_\theta\log\pi_\theta(y_l\mid x)\right].
$$

Pairs the model already separates strongly receive smaller weight. For $\beta=0.5,\Delta=2$, $z=1$, loss is approximately 0.313, and $dL/d\Delta\approx-0.1345$.

Increasing a winner-loser difference does not guarantee the winner's absolute probability increases; both probabilities can fall while the loser's falls more, with mass transferred to other responses. Inspect absolute scores as well as margins.

#### 21.4 What the derivation assumes

The algebra solves a distribution-space optimum with a specified reward model of preferences. It does not prove that finite noisy pair data identifies the reward everywhere, that a neural optimizer reaches the optimum, or that the learned policy remains good outside pair coverage.

Beta is a regularization coefficient in the underlying construction and also scales the pair-classification logit. Its practical effect depends on optimization, data, and reference choice. Avoid a universal rule that larger beta always causes a particular measured KL change in a finite run.

#### 21.5 An exactly solvable preference problem

Let the reference assign equal probability to responses A and B. Suppose the true preference process chooses A over B with probability $p^*=0.8$. Under the DPO model,

$$
\sigma\left(\beta\log\frac{\pi(A)}{\pi(B)}\right)=0.8.
$$

Taking logits gives
$\beta\log(\pi(A)/\pi(B))=\log 4$.
Thus $\pi(A)=\sigma(\log4/\beta)$. At $\beta=1$, the optimum is 0.8; at $\beta=0.5$, it is $16/17\approx0.941$.

This population example holds the preference distribution fixed and uses an unrestricted two-response policy. It helps explain the role of beta in the idealized derivation. Finite noisy pairs, more responses, neural parameter sharing, and optimization can produce a more complicated empirical relationship.

If every observed label favors A, unregularized logistic fitting can drive the classification margin arbitrarily large. Finite-target objectives, early stopping, noisy-label modeling, or additional regularization change that behavior. The data model matters as much as the convenient closed-form derivation.

#### Lab 9 [A; optional B/C] — DPO

**Objective:** derive the loss, implement it, and compare with SFT.

1. Use finite answer logits and the preference pairs from Lab 4.
2. Compute chosen and rejected reference log probabilities.
3. Implement stable logistic loss and compare analytic gradients with finite differences.
4. Compare DPO with SFT on only chosen responses.
5. Track chosen/rejected absolute likelihood, their margin, true reward, and KL.
6. Remove all comparisons involving one response and inspect what remains unidentified.

**Submit:** derivation, numerical gradient agreement, and a coverage-failure example. **B/C extension:** use the tiny instruction LM with completion-only log-probability sums.

<a id="chapter-22"></a>
### Chapter 22 — The preference-optimization family

#### 22.1 Compare targets before names

The family varies feedback type, score definition, reference use, loss shape, and data refresh. A reference-free objective removes a particular reference computation; it does not remove the effects of initialization or regularization.

| Method | Core idea | What to inspect in a reproduction |
|---|---|---|
| **IPO** | Squared target for a preference log-ratio difference | Target margin, beta convention, normalization |
| **KTO** | Desirable/undesirable examples with a prospect-inspired utility | Reference point, class weights, unpaired-data construction |
| **ORPO** | SFT plus an odds-ratio preference term | Definition of response probability and mixing weight |
| **SimPO** | Length-normalized log likelihood as reward, with a margin | Sequence averaging, beta, target margin |
| **CPO** | Contrastive preference learning without the same explicit reference term | SFT regularization and translation/task setting |
| **BCO** | Binary classification through implicit reward scores | Class balance and reward shift/baseline |
| **NCA** | Noise-contrastive learning using explicit reward information | Sampling/noise distribution and reward weights |
| **SLiC-HF** | Calibrate likelihood ranking, with a supervised regularizer | Ranking margin and regularization samples |
| **RRHF** | Rank sampled responses using likelihood scores and feedback | Candidate set, score normalization, ranking loss |
| **Online DPO** | Recollect preference pairs during training | Collection policy, judge, refresh frequency |
| **Nash-style methods** | Optimize against pairwise preferences in a game | Opponent distribution, equilibrium concept, regularization |

Primary readings: [IPO][ipo], [KTO][kto], [ORPO][orpo], [SimPO][simpo], [CPO][cpo], [BCO][bco], [NCA][nca], [SLiC-HF][slic], [RRHF][rrhf], and [Nash learning][nash].

#### 22.2 Three contrasting objectives

Using the DPO log-ratio difference $h=s_w-s_l$, a common IPO convention minimizes

$$
L_{\mathrm{IPO}}=\left(h-\frac{1}{2\beta}\right)^2.
$$

Unlike a separable logistic objective that keeps favoring larger margins, this gives a finite target. Check the exact coefficient convention in the source.

For SimPO, let $\ell(y)=|y|^{-1}\log\pi_\theta(y\mid x)$. A margin-based form is

$$
L_{\mathrm{SimPO}}
=-\log\sigma\left(\beta[\ell(y_w)-\ell(y_l)]-m\right).
$$

This changes both the reference treatment and length weighting. A clean comparison cannot attribute every difference to “no reference” alone.

KTO handles individually labeled desirable and undesirable responses. With $r_\theta=\beta\log(\pi_\theta/\pi_{\mathrm{ref}})$ and a detached KL-derived reference point $z_0$, a pedagogical form uses losses proportional to

$$
\begin{cases}
1-\sigma(r_\theta-z_0), & \text{desirable},\\
1-\sigma(z_0-r_\theta), & \text{undesirable}.
\end{cases}
$$

The reference-point estimator and class weights are essential parts of a faithful implementation. Label this as a KTO-style experiment unless they match the chosen paper version.

ORPO combines chosen-response likelihood with a preference term built from response odds. If the sequence score is a geometric-mean token probability $p$, its log odds is $\log p-\log(1-p)$. Replacing that with a raw sum of token log probabilities changes the construction.

#### 22.3 Rankings, binary classifiers, and games

SLiC-HF and RRHF connect feedback ranking with likelihood calibration, while maintaining supervised pressure to produce useful text. Their candidate-generation choices determine which alternatives the policy learns to order.

BCO frames feedback as classification of individual responses via implicit reward scores. NCA uses explicit reward information and a contrastive distributional formulation. Treat neither as simply “DPO with another name.”

Scalar reward models induce a transitive ordering for a fixed prompt. Real pairwise preferences can cycle: A beats B, B beats C, C beats A. Nash-style preference learning studies policies against opponents in such preference games; a mixed strategy can matter when no single response beats all others.

#### Lab 10 [A; optional B/C] — KTO/SimPO-style objectives

**Objective:** change the objective and compare gradients.

1. Hold initial logits and data fixed; calculate DPO, IPO, and SimPO losses.
2. Create separate desirable/undesirable labels for a KTO-style experiment.
3. Document which labels were genuinely given and which were derived.
4. Plot loss and gradient versus margin.
5. Add length imbalance, label noise, and an unobserved response.
6. Compare equal update and equal data budgets; report all hyperparameters.

**Submit:** objective equations, gradient plots, and a table of confounders. **Check:** do not call a simplified loss a faithful reproduction of a named algorithm. **B/C extension:** run the same comparison on a small LM after matching loss masks.

<a id="chapter-23"></a>
### Chapter 23 — When preference optimization works

#### 23.1 Coverage controls what can be learned

Offline pairs can teach distinctions among available responses. They provide little direct evidence about entirely unobserved regions. Saturation can mean the model has learned all easy pair distinctions, the data lacks stronger candidates, or evaluation has hit a ceiling.

Distribution shift occurs when the updated policy generates responses unlike the training pairs. Iterative DPO recollects pairs from newer policies, potentially improving coverage. It can also amplify judge biases or collapse diversity.

Self-generated pairs need an independent basis for deciding which response is better. A policy repeatedly preferring its own style is not sufficient evidence of task improvement.

#### 23.2 Choose objectives through diagnosis

| Situation | Useful first experiment |
|---|---|
| Excellent demonstrations, little preference data | SFT baseline |
| Reliable pairs covering deployment behavior | Offline preference objective |
| Executable reward and ability to collect rollouts | RL or online preference collection |
| Useful individual accept/reject labels | Unpaired-feedback objective |
| Strong model but weak judge | Improve feedback validation first |

This is an experimental starting point, not a universal ranking. Compare with SFT and a fixed inference baseline at matched resources. Record training compute, labeling cost, and generation tokens separately.

**Exercise.** A DPO model improves held-out pair accuracy but loses answer accuracy. List three hypotheses: preference labels reward the wrong attribute, the pair test lacks deployment coverage, or probability mass moved to unseen poor responses. Design one discriminating measurement for each.

**Part VI mastery check:** reconstruct DPO without consulting the formula, then explain why its derivation does not eliminate coverage, noise, or optimization problems.

<a id="part-vii"></a>
## Part VII — Modern Policy Optimization for LLMs

**Chapter navigation:** [24 · REINFORCE comes back](#chapter-24) · [25 · GRPO](#chapter-25) · [26 · Beyond vanilla GRPO](#chapter-26) · [27 · Understanding policy-gradient design choices](#chapter-27)

**Learning goals:** compare critic-free estimators and identify the distributional effect of each normalization.

<a id="chapter-24"></a>
### Chapter 24 — REINFORCE comes back

#### 24.1 Sequence rewards can simplify the estimator

When a complete response receives one reward, the whole response can be treated as an action in a contextual bandit. The score-function gradient uses the sum of token log probabilities:

$$
\hat g=\frac1N\sum_i
(R_i-b_i)\nabla_\theta\log\pi_\theta(y_i\mid x_i).
$$

A critic is optional. It is useful when its variance reduction justifies its estimation error and cost. Long language-model responses make learning a token-level critic demanding, while multiple complete samples can supply a simpler baseline. [Revisiting REINFORCE for RLHF][rloo]

#### 24.2 Leave-one-out baselines

Sample $G>1$ independent responses for the same prompt. Define

$$
b_i=\frac{1}{G-1}\sum_{j\neq i}R_j,\qquad
\hat A_i=R_i-b_i.
$$

Conditional on the prompt, $b_i$ is independent of sample $i$ under independent sampling. Thus it is a valid action-independent baseline for that sample. Correlated generation schemes require a fresh argument.

For rewards $(0,1,1)$, the leave-one-out advantages are $(-1,0.5,0.5)$. The negative response is penalized relative to the alternatives, and the positives are reinforced.

ReMax uses the reward of a greedy response as a prompt-conditioned baseline in its construction. If that baseline is detached and does not depend on the sampled response, it satisfies the relevant baseline independence condition. Its usefulness depends on correlation with the sampled return. [ReMax][remax]

#### Lab 6 [A; optional B/C] — REINFORCE for text

**Objective:** train on an artificial sequence task.

1. Start with binary strings of length four, rewarding an explicit rule such as exactly three ones.
2. Enumerate all 16 strings to compute expected reward and an exact gradient.
3. Implement sampled REINFORCE with summed token log probabilities.
4. Compare no baseline, a fixed baseline, and an independently estimated mean.
5. Check Monte Carlo gradient means against the exact result.
6. Transfer to short character responses with an executable reward.

**Submit:** exact-gradient comparison, variance by batch size, and learning curves. **Check:** the reward and baseline are detached in the actor update. **B/C extension:** use the same rule in a small LM's output format.

#### Lab 11 [A; optional B/C] — RLOO

**Objective:** sample several responses and use leave-one-out baselines.

Use groups of 2, 4, and 8 on the same tasks. Match total sampled responses across conditions. Estimate gradient bias and variance at fixed policy parameters before comparing training curves. Verify the hand calculation above.

**Submit:** a group-size table with mean gradient error, variance, sample count, and runtime. **Check:** each baseline excludes its own response; $G=1$ is rejected. **Research extension:** create correlated groups and test whether the original unbiasedness argument still applies.

<a id="chapter-25"></a>
### Chapter 25 — GRPO

#### 25.1 Relative advantages within a prompt

A common outcome-reward GRPO construction samples a group and sets

$$
\bar R=\frac1G\sum_iR_i,\qquad
s_R=\sqrt{\frac1G\sum_i(R_i-\bar R)^2},\qquad
\hat A_i=\frac{R_i-\bar R}{s_R+\varepsilon_{\mathrm{num}}}.
$$

Some implementations use a different standard-deviation convention. Record it. The numerical epsilon is distinct from the clipping epsilon.

For rewards $(0,0,1,1)$, the population standard deviation is 0.5 and advantages are approximately $(-1,-1,1,1)$. If all rewards are equal, the centered advantages are zero; a KL or entropy term may still update the policy, but outcome-relative learning has no signal from that group.

#### 25.2 The clipped surrogate

A representative GRPO surrogate is

$$
J=
\frac1G\sum_i\frac1{T_i}\sum_t
\left[
\min\{\rho_{it}\hat A_i,
\text{clip}(\rho_{it},1-\epsilon,1+\epsilon)\hat A_i\}
-\beta\hat k_{it}
\right].
$$

Here the ratio is token-level, advantages are response-level, and the outer reduction averages per response. All three choices matter. The original method also discusses process-reward variants. [DeepSeekMath][grpo]

A frequently used nonnegative sample KL expression is $u-1-\log u$, where $u=\pi_{\mathrm{ref}}(a\mid h)/\pi_\theta(a\mid h)$. Under actions sampled from the current policy at a fixed history and compatible support, its expectation equals policy-to-reference KL. On old-policy samples, or with particular autodiff treatment, the value and gradient interpretation needs additional care. A convenient positive number is not automatically an unbiased KL gradient.

#### 25.3 Why normalization is not innocuous

The group mean includes the sample being updated. For independent samples at one prompt,

$$
\mathbb E[(R_i-\bar R)\nabla\log\pi(y_i)]
=\frac{G-1}{G}\nabla\mathbb E[R].
$$

Cross-sample score terms vanish, but the self-term subtracts $1/G$ of the gradient. With fixed group size this is a scalar factor before other modifications. Dividing by sample standard deviation introduces a further random, reward-dependent weighting and can change relative prompt contributions.

Per-response division by $T_i$ also changes the score-function weighting when length depends on actions. Group normalization, token clipping, and length averaging should each be studied separately.

> **Figure placeholder F04 — Group-relative learning.** Show three prompt groups: mixed rewards, all failures, and all successes. Display centered rewards and normalized advantages. Caption: “Relative feedback depends on the sampled group, not just a response's absolute reward.”

**Checkpoint.** Why can increasing group size help even when the loss formula is unchanged? It changes the probability of observing mixed outcomes and the noise of group statistics.

**Lab placement:** Lab 12 follows Chapter 28, after you build the correctness checker it requires.

#### 25.4 Exact expectation for a two-sample binary group

Take a Bernoulli policy with success probability $p$, reward equal to the sampled action, and group size two. With population-standard-deviation normalization and negligible numerical epsilon, an all-equal group produces zero advantages.

A mixed group has advantages $+1$ for the successful action and $-1$ for the failed action. Their logit scores are $1-p$ and $-p$. The group-average gradient is therefore

$$
\frac12[(1-p)-(-p)]=\frac12.
$$

Mixed groups occur with probability $2p(1-p)$, so the expected standardized-group gradient is $p(1-p)$, which happens to equal the true gradient in this special case.

This does not establish general unbiasedness. Different group sizes, multivalued rewards, policies with several actions, nonzero epsilon, or response-length weighting can break the coincidence.

**Research lesson:** a passing two-action test can be a necessary sanity check without being a sufficient estimator proof. Include a three-action reward distribution and variable-length responses before making a general claim.

<a id="chapter-26"></a>
### Chapter 26 — Beyond vanilla GRPO

#### 26.1 Separate algorithm changes from training recipes

“Beyond GRPO” can mean a changed gradient estimator, reduction, sampling distribution, regularizer, or systems implementation. Improvements from a bundle cannot identify which component caused them.

| Method or design | Main distinction to study | A controlled question |
|---|---|---|
| Dr. GRPO | Removes response-length and reward-standard-deviation normalizers from its proposed form | Does observed length behavior change with all other choices fixed? |
| REINFORCE++ | Critic-free optimization with stabilizing choices including global advantage normalization | How does batch-wide scaling differ from per-prompt scaling? |
| DAPO | Asymmetric clipping, dynamic sampling, token-level reduction, and overlong handling | Which intervention accounts for gains at equal rollout budget? |
| GSPO | A length-normalized sequence likelihood ratio and sequence-level clipping | How do token outliers affect an entire response's update? |

Read the chosen versions of [Dr. GRPO][drgrpo], [REINFORCE++][reinforcepp], [DAPO][dapo], and [GSPO][gspo]. These names cover specific published formulations; implementation variants can differ.

#### 26.2 Ratios and clipping units

The full sequence importance ratio is

$$
\rho_i^{\mathrm{seq}}
=\exp\left(\sum_t\log\frac{\pi_\theta(y_{it}\mid h_{it})}
{\pi_{\mathrm{old}}(y_{it}\mid h_{it})}\right).
$$

A length-normalized sequence ratio used in GSPO is its geometric mean:

$$
s_i=\exp\left(\frac1{T_i}\sum_t
\log\frac{\pi_\theta(y_{it}\mid h_{it})}
{\pi_{\mathrm{old}}(y_{it}\mid h_{it})}\right).
$$

The latter is a useful surrogate quantity, not the full Radon–Nikodym ratio for exact sequence-level importance sampling. If two token ratios are 2 and 0.5, the sequence ratio and geometric mean are both 1, even though token-level clipping can treat the individual tokens very differently.

Asymmetric clipping uses interval $[1-\epsilon_{\mathrm{low}},1+\epsilon_{\mathrm{high}}]$. A higher upper threshold changes how quickly positively advantaged sampled actions can be reinforced before the surrogate flattens.

#### 26.3 Selection and length handling

Dynamic sampling can discard groups with identical rewards and collect more informative groups. This changes the effective prompt distribution and consumes extra rollouts. Count discarded generations when comparing compute.

Overlong responses may be unfinished correct attempts, loops, or deliberate exploitation. Treating every truncation as an ordinary incorrect final answer introduces a particular signal. Soft penalties or filtering are alternative definitions, each requiring a declared task budget.

Removing certain normalizers does not make every remaining estimator universally unbiased: clipping, stale trajectories, group centering, and sampling selection still need analysis.

**Exercise.** Design a factorial experiment varying only standard-deviation normalization and response-length normalization. Predict which statistics would distinguish prompt reweighting from length effects.

<a id="chapter-27"></a>
### Chapter 27 — Understanding policy-gradient design choices

#### 27.1 Audit the estimator, line by line

For every trainer, fill in this table before running it:

| Component | Required answer |
|---|---|
| Prompt distribution | Original, weighted, filtered, or adaptively selected? |
| Behavior distribution | Which weights and decoding transformations generated samples? |
| Reward unit | Token, step, response, or trajectory? |
| Baseline | Learned, independent, greedy, group mean, or leave-one-out? |
| Advantage scaling | None, batch, group, or running statistics? |
| Ratio | Token, full sequence, or normalized sequence? |
| Clipping | Which unit and which asymmetric/symmetric bounds? |
| Loss reduction | Sum, sequence average, or batch-token average? |
| KL/entropy | Direction, sampling distribution, estimator, coefficient? |
| Update reuse | Number of epochs and policy staleness? |

#### 27.2 Denominators define weighting

For token contribution $u_{it}$, compare

$$
\frac1N\sum_i\frac1{T_i}\sum_tu_{it},
\qquad
\frac{\sum_{i,t}u_{it}}{\sum_iT_i},
\qquad
\frac1{NC}\sum_{i,t}u_{it}.
$$

The first gives equal response weight after averaging each response. The second gives equal token weight within a realized batch. The third divides by fixed constant $C$, preserving the relative sum of each response's token contributions. Random denominators can affect expectations, and changing batch construction can change their behavior.

**Worked example.** Two responses have 2 and 8 tokens with identical per-token contribution 1. All three formulas can look similar in a single aggregate number, yet their per-token gradient coefficients are $1/4$ versus $1/16$ in the first, $1/10$ for all tokens in the second, and $1/(2C)$ in the third.

**Research checkpoint.** State the exact target before using the word “bias.” A biased estimator of unregularized expected reward may be an intentional surrogate for another objective.

**Part VII deliverable:** an estimator card comparing REINFORCE, RLOO, GRPO, and one variant under common notation. The controlled training comparison appears in Lab 13 after experimental design.

#### 27.3 Prompt weighting can change even without explicit weights

Consider two prompt families with the same number of samples but different reward scales. In family A, useful responses differ by reward 0.1; in family B, they differ by 10. Raw policy gradients reflect this scale difference. Standardizing within each prompt can make the two families contribute more similarly.

Whether this is desirable depends on the intended objective. If reward scale encodes real utility, standardization can erase it. If scale is an arbitrary artifact of two graders, standardization may correct an undesirable imbalance. The same mathematical operation can be helpful or harmful under different specifications.

Selection creates another implicit weight. If one family frequently produces flat groups that are discarded, its contribution changes with the current policy. A nominally uniform prompt sampler no longer guarantees a uniform training objective.

**Exercise:** construct two finite tasks with independently adjustable reward scale and success rate. Measure their gradient contributions under raw rewards, batch normalization, group normalization, and flat-group filtering.

<a id="part-viii"></a>
## Part VIII — Reinforcement Learning with Verifiable Rewards

**Chapter navigation:** [28 · RLVR](#chapter-28) · [29 · Reasoning emergence](#chapter-29) · [30 · Reasoning-model training pipelines](#chapter-30) · [31 · Reward design for reasoning](#chapter-31)

**Learning goals:** build a trustworthy executable signal and interpret claims about reasoning improvement.

<a id="chapter-28"></a>
### Chapter 28 — RLVR

#### 28.1 Verification is a task definition

RLVR uses a computable outcome: a math answer, unit-test result, proof-checker acceptance, or structured-output constraint. It removes some judgment costs, but the checker still defines a proxy.

Exact string match may reject equivalent answers such as “1/2” and “0.5.” Symbolic equivalence can recognize more cases but requires domains and assumptions: $x/x=1$ excludes $x=0$. Numerical comparison requires tolerances. A formal proof checker verifies a proof relative to its formal specification and trusted components; it does not ensure the specification matches the intended informal problem.

For code, passing supplied tests demonstrates agreement on those tests. It does not imply complete correctness. Partial credit can provide a denser signal, but passing easy tests repeatedly may become an attractive local optimum.

#### 28.2 Separate parser, evaluator, and reward

The parser extracts a well-defined candidate. The evaluator determines correctness. The reward maps evaluation outcomes to a number. Store all three results. Otherwise parser failures, genuine mistakes, and timeouts become indistinguishable zeros.

**Example.** A response containing both a wrong and a correct answer should not automatically pass because a loose regex finds the correct number somewhere. Define exactly which field is graded.

#### 28.3 Distinguish correctness, coverage, and exploitability

A verifier's false-positive rate measures acceptance of wrong solutions. Its false-negative rate measures rejection of correct ones. Both rates depend on the test distribution. Optimization can move outputs into a region where false positives are much more common.

For example, a checker tested on ordinary arithmetic mistakes may reject all of them, yet accept a response containing multiple contradictory answers because of its parsing rule. The ordinary test set did not measure that failure mode.

Build verifier tests from three sources: normal correct/incorrect examples, boundary cases derived from the specification, and adversarial examples selected to exploit the implementation. Report them separately.

In code tasks, mutation testing offers one useful idea: deliberately change correct programs in ways that should break behavior, then see which mutants the tests fail to catch. Surviving mutants reveal gaps in test coverage. The same principle applies to mathematical transformations and structured outputs.

#### Lab 14 [A] — Verifier

**Objective:** build a math or code correctness checker.

1. Define a strict response schema for small arithmetic tasks.
2. Parse exactly one answer; handle malformed, missing, and multiple answers explicitly.
3. Use integer arithmetic or exact rational arithmetic for ground truth.
4. Test equivalent forms, extra text, very large inputs, and boundary cases.
5. Record parser status, correctness, reward, and checker version.
6. Create a hidden audit set of tricky cases before training.

**Submit:** checker, input/output contract, and adversarial test cases. **Check:** never execute arbitrary response text as an unrestricted expression.

#### Lab 12 [A; optional B/C] — GRPO

**Objective:** train on a tiny automatically checked math task.

1. Use a finite answer policy over small integer answers, or the tiny LM.
2. Sample four or eight responses per prompt and score them with Lab 14.
3. Implement detached group advantages, old-policy ratios, and clipping.
4. Handle all-equal reward groups explicitly.
5. Compare summed and per-response-averaged token contributions.
6. Record reward, entropy, KL if used, group variance, response length, and discarded groups.

**Submit:** objective specification, one printed group calculation, held-out accuracy, and an estimator analysis. **Checks:** ratios start at 1, padding is excluded, and groups are not mixed across prompts.

#### Lab 15 [A; optional B/C] — Miniature RLVR

**Objective:** improve measured performance on a verifiable task.

Freeze a task split and evaluation budget. Compare the initial policy, SFT, and RLVR using the checker. Hold out operand combinations or task rules that require generalization. Use at least three seeds for an initial variance estimate; report individual runs.

**Submit:** learning curves, uncertainty, checker-error audit, and an explicit statement of what generalized. **Expected behavior:** a simple policy should learn a discoverable reward rule, but improvement is an experimental result to establish. **B/C extension:** increase linguistic variation while preserving the same mathematical task.

<a id="chapter-29"></a>
### Chapter 29 — Reasoning emergence

#### 29.1 Observable traces and latent computation

A longer reasoning trace is observable text. It may implement useful decomposition, search, reflection, backtracking, or verification. It can also repeat phrases without causal value. A model saying “wait” is not by itself evidence of a new reasoning mechanism.

Outcome reward can reinforce successful intermediate strategies without step labels. Whether this discovers a new strategy, increases an existing strategy's probability, or exploits evaluation artifacts is an empirical question.

#### 29.2 Evidence for a mechanism

Measure task accuracy under controlled inference budgets. Compare pass@1 and multi-sample success. Inspect error types and response length. Test interventions: remove a claimed verification step, provide a corrupted intermediate result, or change the problem representation.

If pass@1 rises while sufficiently sampled pass@k stays similar, one plausible explanation is redistribution toward already accessible correct responses. This pattern does not prove there was no capability change; finite sampling and task coverage limit the inference.

**Worked example.** Model A uses 50 tokens and solves 60 tasks; model B uses 500 and solves 70. To isolate a training gain, compare both at fixed budgets and also show the accuracy/compute curve.

**Exercise.** Form two competing hypotheses for longer traces: useful search versus reward-favored verbosity. State a perturbation and expected observations for each.

<a id="chapter-30"></a>
### Chapter 30 — Reasoning-model training pipelines

#### 30.1 A sequence of distributions

A reasoning pipeline can include cold-start SFT, an RL stage, rejection sampling, a second SFT phase, general alignment, and distillation. Each stage changes the policy and the distribution of data available to later stages. The [DeepSeek-R1 report][r1] is a concrete primary case study, not a universal recipe.

Cold-start SFT can teach a usable response format and seed successful trajectories. RL then explores and reinforces rewarded behavior. Rejection sampling filters a larger candidate pool, and another SFT phase absorbs accepted traces. General-alignment training can recover instruction following or other abilities that narrowed during task-specific optimization.

#### 30.2 Trace filtering and task mixtures

Filter for outcome correctness, readability, unsupported steps, repetition, and suitable length. A trace can pass final-answer checking while containing a false explanation. Keep filtering reasons and acceptance rates by task.

General capability preservation requires evaluation outside the specialized reasoning domain. A model can improve arithmetic and degrade ordinary instruction following. Task mixtures and retained demonstrations provide interventions, but the balance is experimental.

**Exercise.** Draw a data lineage table for a five-stage pipeline. For every dataset, identify generator checkpoint, selection rule, labeler/verifier version, and downstream consumer.

<a id="chapter-31"></a>
### Chapter 31 — Reward design for reasoning

#### 31.1 Multi-objective rewards

A reward can combine correctness, format, process quality, length, efficiency, and style:

$$
R=w_cR_{\mathrm{correct}}+w_fR_{\mathrm{format}}
+w_pR_{\mathrm{process}}-w_\ell C_{\mathrm{length}}
-w_tC_{\mathrm{tools}}.
$$

Weights define tradeoffs. If format reward is easier than correctness, the policy can become beautifully formatted and wrong. If length penalty exceeds the benefit of a difficult correct solution, it can learn to stop early.

Lexicographic or constrained designs can encode priorities more directly: maximize correctness subject to a token budget, or compare efficiency only among correct outputs. Those formulations still require a practical optimization method.

#### 31.2 Curricula and anti-hacking checks

Begin with tasks where success is observable often enough to learn, then vary difficulty. Keep a fixed evaluation set to distinguish a changing task mixture from a changing policy.

Audit reward components independently. Report accuracy, validity, length, and tool cost, not just their weighted sum. A scalar improvement can conceal a regression in the primary goal.

**Numerical example.** Correctness reward 1 minus 0.02 per step makes a 60-step correct solution worse than an immediate zero-reward stop unless the task's stopping penalty changes that comparison. Calculate the implied tradeoff before training.

**Part VIII deliverable:** a versioned verifier and an RLVR experiment with independently audited outcomes.

<a id="part-ix"></a>
## Part IX — Exploration, Search and Self-Improvement

**Chapter navigation:** [32 · Exploration in language-model RL](#chapter-32) · [33 · Best-of-N and rejection sampling](#chapter-33) · [34 · Search-enhanced training](#chapter-34) · [35 · Self-training and self-improvement](#chapter-35)

**Learning goals:** distinguish a sampling policy from a model distribution and understand how search changes both inference and training data.

<a id="chapter-32"></a>
### Chapter 32 — Exploration in language-model RL

#### 32.1 Sampling changes the policy

Temperature sampling uses

$$
\pi_\tau(a\mid h)=
\frac{\exp(z_a/\tau)}{\sum_b\exp(z_b/\tau)}.
$$

For $\tau>1$, logits are flattened; for $0<\tau<1$, they are sharpened. Top-k retains a fixed number of highest-scoring tokens. Top-p retains a set reaching a cumulative probability threshold, then renormalizes. These transformations define the behavior distribution and can alter support.

Exploration concerns finding useful alternatives. High token entropy can generate varied punctuation while preserving the same incorrect strategy. Measure diversity at the level relevant to the task: solution approach, tool choice, or discovered evidence.

#### 32.2 Why difficult prompts can be uninformative

For independent samples with success probability $p$, the probability of at least one success in $G$ attempts is $1-(1-p)^G$. At $p=0.001,G=8$, it is about 0.008. Most groups yield no positive examples.

For binary rewards, a mixed group occurs with probability

$$
1-p^G-(1-p)^G.
$$

This is low for both very easy and very hard prompts. Difficulty-aware sampling can focus on informative tasks, but should preserve coverage or explicitly change the intended objective.

#### 32.3 Curricula and exploration-exploitation

Exploitation samples likely good actions. Exploration spends resources to discover alternatives. A curriculum can vary difficulty, action budgets, or available hints. Adaptive sampling should log selection probabilities and include held-out evaluation on the original target distribution.

**Exercise.** Compare increasing group size with sampling more distinct prompts at a fixed response budget. Which helps within-prompt baseline estimation, and which improves coverage of rare task types?

<a id="chapter-33"></a>
### Chapter 33 — Best-of-N and rejection sampling

#### 33.1 Selection changes the output distribution

Best-of-N generates $N$ candidates and selects the highest-scoring one. With a perfect binary verifier and independent samples, success follows $1-(1-p)^N$. With an imperfect ranker, selection can prefer errors.

If $p=0.2$, five samples have an idealized probability $1-0.8^5\approx0.672$ of including a correct answer. That does not mean the trained policy's pass@1 improved; the gain used more inference.

#### 33.2 Accepted-generation training

Generate responses, keep those meeting a criterion, and train on the survivors. If acceptance probability is $a(y)$, the accepted distribution is

$$
q_{\mathrm{accept}}(y\mid x)=
\frac{\pi(y\mid x)a(x,y)}
{\mathbb E_{y'\sim\pi}[a(x,y')]}.
$$

This equation shows the reweighting. In language-model practice, “rejection sampling” often refers to filtering generated samples, rather than an exact sampler for a specified target density using a known envelope.

Repeated acceptance and SFT can strengthen useful behavior or narrow diversity. Monitor acceptance rate, unique solutions, and the fraction of tasks producing no training examples.

**Exercise.** A filter retains only the shortest correct solution. What distribution does the next SFT stage imitate? Explain why this is a correctness-and-length intervention rather than pure correctness learning.

#### 33.3 Compute-matched comparison of selection and training

Suppose an unchanged policy has pass@1 of 0.2 and a trained policy has pass@1 of 0.4 under the same decoding setup. The unchanged policy's ideal pass@5 is about 0.672, while the trained policy's is $1-0.6^5\approx0.922$, assuming independent trials and a perfect selector.

These numbers answer different questions. The first comparison measures one-sample policy quality. The second measures what each policy can provide with five samples and an oracle. Neither includes the compute spent training the improved policy.

For a deployed system, account for amortization: a training cost may be justified over many future queries. Also account for verifier cost and imperfect selection. A model that generates correct answers frequently but is hard to rank can be less useful than its oracle pass@k suggests.

**Exercise:** draw accuracy against inference samples for both policies, then add a fixed training cost divided across 100, 10,000, and one million queries. Explain why the preferred strategy can depend on expected usage.

<a id="chapter-34"></a>
### Chapter 34 — Search-enhanced training

#### 34.1 Search explores a tree of decisions

Beam search preserves a fixed number of high-scoring partial sequences. It favors model likelihood unless another score is added; high likelihood is not synonymous with high reward.

Tree search expands alternative prefixes or actions and backs up value estimates. A common UCT selection score is

$$
\bar Q(s,a)+c\sqrt{\frac{\log N(s)}{N(s,a)}},
$$

with a rule for unvisited actions. The exploration bonus favors underexplored branches. PUCT additionally incorporates policy priors. MCTS is the repeated selection, expansion, evaluation, and backup procedure; a formula alone does not specify the whole algorithm.

Process rewards and learned values can guide expansion. If a value model overestimates a branch, search can allocate increasing resources to that error.

#### 34.2 Distilling search

Search-generated supervision can include winning trajectories, action visit distributions, or value targets. Distillation moves some inference-time computation into the policy's parameters.

Training on only winners hides failed branches and can create coverage gaps. The training dataset is selected by the search procedure, so a policy-gradient interpretation requires appropriate sampling analysis. Supervised distillation is often a simpler and more honest description.

**Worked example.** A calculator agent has four candidate operations at each of three steps. Exhaustive search has up to $4^3=64$ action sequences. A width-two beam examines fewer paths but can discard the only successful prefix if its early score is poor.

> **Figure placeholder F05 — Search and distillation.** Draw a small tree with explored failures, a successful path, and value estimates. Show which records become SFT and value targets. Caption: “Search chooses both what to execute and what data the next policy sees.”

**Exercise.** Compare a stronger policy with no search against a weaker policy with search at equal wall time and equal model tokens. Explain why the two comparisons answer different questions.

<a id="chapter-35"></a>
### Chapter 35 — Self-training and self-improvement

#### 35.1 A feedback loop needs new information

Self-generated tasks, answers, critiques, and rewards can expand data. The useful information may come from an executable environment, an independent judge, search, a stronger teacher, or a curriculum that exposes previously rare behavior.

Repeatedly training on unfiltered model samples can preserve errors or concentrate mass on common modes. Finite sampling can lose rare behaviors. This motivates tracking diversity and maintaining external or retained data.

Self-critique and iterative refinement can improve individual responses if the critic identifies correctable errors. A model can also confidently “correct” a right answer into a wrong one. Measure transition counts between correct and incorrect states.

#### 35.2 Bootstrapping and self-play

In self-play, policies generate challenges or opponents for each other. Learning can produce a useful curriculum, cycling behavior, or exploitation of opponent-specific weaknesses. Evaluate against held-out opponents and fixed tasks.

An automatic task generator can maximize learner failure, but impossible tasks are not useful supervision. Learning progress or a target difficulty range can produce a more informative curriculum.

**Worked example.** Refinement turns 20 wrong answers into right ones but turns 15 right answers into wrong ones. Reporting only corrected examples exaggerates its benefit; net accuracy gains by five cases.

**Part IX deliverable:** a comparison of direct sampling, selection, search, and distilled behavior, with all generation costs counted.

<a id="part-x"></a>
## Part X — Agent Foundations

**Chapter navigation:** [36 · What makes an LLM agent](#chapter-36) · [37 · Agents as MDPs and POMDPs](#chapter-37) · [38 · Agent architectures](#chapter-38)

**Learning goals:** define the agent's decision process and build a local environment with explicit state, observations, actions, and termination.

<a id="chapter-36"></a>
### Chapter 36 — What makes an LLM agent

#### 36.1 Model and harness together produce behavior

The model proposes actions. A harness formats observations, exposes tools, validates calls, manages memory, executes actions, and decides when to stop. The environment changes in response.

A model with identical weights can behave differently under different tool descriptions or retry rules. Therefore “the agent checkpoint” is insufficient to reproduce the agent.

| Component | Example | Failure to inspect |
|---|---|---|
| Policy | LM choosing a database query | Invalid arguments |
| Harness | Parses JSON and enforces budgets | Silent retries alter trajectories |
| Tool | Calculator or search API | Error/result ambiguity |
| Environment | Documents, database, repository | State contamination between episodes |
| Memory | Prior observations or summary | Lost constraints |
| Termination | Submit, failure, timeout | Incorrect success detection |

#### 36.2 Actions and outcomes

An action can be free text, a structured call, or a sequence of calls. Decide whether reasoning tokens are part of the policy action and whether they incur cost. Tool observations are generated by the environment, not chosen by the actor.

Terminal rewards can assess the final answer or world state. Intermediate rewards can assess progress, but may change incentives.

**Exercise.** Specify a “find a record and compute a total” agent. List available tools, hidden data, success condition, maximum calls, and what happens when a call is malformed.

<a id="chapter-37"></a>
### Chapter 37 — Agents as MDPs and POMDPs

#### 37.1 World state versus policy observation

Let $s_t$ be the full environment state, $o_t$ an observation, and $a_t$ an action. A partially observed policy conditions on history $h_t=(o_0,a_0,\ldots,o_t)$:

$$
\pi_\theta(a_t\mid h_t),\qquad
s_{t+1}\sim P(\cdot\mid s_t,a_t),\qquad
o_{t+1}\sim O(\cdot\mid s_{t+1}).
$$

The trajectory distribution includes policy, transition, and observation factors. If only policy factors depend on $\theta$, the score-function derivative sums only their log probabilities.

A tool call can change future information even when it does not change the underlying world. Querying the right document changes what the policy can infer.

#### 37.2 Memory is state estimation

A finite history window discards old information. A summary compresses it. Either can merge histories requiring different actions, a problem called state aliasing.

Suppose two conversations have identical latest messages but one earlier authorized a write and the other did not. A summary that drops that distinction creates an inadequate observation state.

Belief updates formally combine prior belief, action, and new observation. Most language agents use textual memory instead of an explicit posterior, but the conceptual requirement remains: preserve information needed for the next decision.

**Checkpoint.** When is an observation-only policy Markov? When that observation is sufficient to predict future task-relevant transitions and rewards given the action. A screenshot or last message rarely guarantees this.

#### 37.3 A two-state belief update

Suppose a database replica is either fresh or stale. Before querying, assign probability 0.7 to fresh and 0.3 to stale. A diagnostic observation is “version current,” with likelihood 0.9 under fresh and 0.2 under stale.

Bayes' rule gives

$$
\Pr(\mathrm{fresh}\mid\mathrm{current})=
\frac{0.9(0.7)}{0.9(0.7)+0.2(0.3)}
\approx0.913.
$$

The observation improves confidence but does not prove freshness. A memory summary that stores “replica is definitely fresh” discards uncertainty and can cause overconfident actions later.

A language agent rarely maintains such a small explicit posterior, but the example clarifies what useful memory must preserve: evidence, its source, and remaining uncertainty. State compression should be evaluated by downstream decision quality, not only by summary readability.

<a id="chapter-38"></a>
### Chapter 38 — Agent architectures

#### 38.1 Ways to organize decisions

ReAct interleaves reasoning and actions with observations. A planner/executor splits higher-level planning from execution. Reflection revises a plan or memory after feedback. Critic or verifier agents assess proposed behavior. Hierarchical policies choose subgoals and lower-level actions. Multi-agent systems assign roles or parallel investigations. [ReAct][react]

These structures change information flow, cost, and failure recovery. Extra agents may provide independent views or simply repeat correlated errors.

#### 38.2 Design the smallest adequate harness

Expose a compact action schema, explicit tool errors, a bounded call budget, and a final-answer action. Record every transition. Avoid giving the policy a hidden success label in its observation unless that is part of the intended task.

If planning text is generated but never affects execution, it is not evidence that planning caused success. Compare ablations with and without the planning stage at matched budgets.

#### Lab 18 [A; optional B/C] — Tool-use environment

**Objective:** connect an LM to a calculator, search index, or database and feed observations back.

**A setup:** create a local table with fictional records and a calculator supporting only add/subtract/multiply. A finite categorical policy over structured actions is sufficient initially; the tiny LM can then generate those actions.

1. Define reset(seed), observe(), step(action), and termination behavior.
2. Create 50 tasks requiring one or two lookups and a calculation.
3. Parse actions with a strict schema; return structured success/error observations.
4. Implement a rule-based oracle policy for checking task solvability.
5. Run random and oracle policies to establish lower and upper baselines.
6. Attach the tiny LM, preserving all actions and tool results in the context.

**Submit:** environment contract, three complete trajectories, success metrics, and an error taxonomy. **Checks:** reset isolation, deterministic replay under a fixed seed, no hidden answers in observations, and a hard step budget. **B/C extension:** substitute a small pretrained LM while keeping the environment unchanged.

**Part X mastery check:** state precisely which behavior belongs to the policy, which belongs to the harness, and which belongs to the environment.

<a id="part-xi"></a>
## Part XI — Agent Post-Training

**Chapter navigation:** [39 · Agent trajectory datasets](#chapter-39) · [40 · SFT for tool use](#chapter-40) · [41 · RL for agents](#chapter-41) · [42 · Credit assignment for long-horizon agents](#chapter-42) · [43 · Harnessed agentic RL](#chapter-43) · [44 · Training web and search agents](#chapter-44) · [45 · Training coding agents](#chapter-45) · [46 · Computer-use agents](#chapter-46) · [47 · Multi-agent post-training](#chapter-47)

**Learning goals:** convert interaction logs into valid training samples, assign credit across decisions, and evaluate specialized agent environments.

<a id="chapter-39"></a>
### Chapter 39 — Agent trajectory datasets

#### 39.1 A trace is a sequence of causally ordered events

An agent dataset should preserve what the policy knew before every action. A trace containing the final successful answer before an earlier decision creates hindsight leakage.

Use separate event types for user input, model output, tool request, tool result, environment update, and terminal evaluation. Store episode and step IDs, policy version, action probabilities when available, model-visible context, and outcome.

```json
{
  "episode_id": "task_017_seed_2",
  "step": 3,
  "policy_version": "policy_004",
  "event": "model_action",
  "observation_ids": ["obs_001", "obs_002"],
  "action": {"tool": "lookup", "key": "record_8"},
  "termination": false
}
```

The corresponding tool-result record should refer to this action and store its actual return. Large payloads can be content-addressed, provided the referenced bytes remain available.

#### 39.2 Replay, filtering, and compression

Replay reconstructs an episode against the same environment version. Filtering can select success, informative failure, or rare states. Success-only filtering improves apparent quality while hiding recovery.

Compression removes repetitive content or stores summaries. Preserve the original trace alongside the compressed training view. Otherwise a later debugging session cannot distinguish a model mistake from information removed by preprocessing.

Action masks restrict allowed actions. If used during sampling, the masked and renormalized policy is the behavior policy. Save enough information to reproduce it.

#### Lab 19 [A] — Agent trajectory collection

**Objective:** record states, actions, tool results, and outcomes.

Run random, oracle, and model policies in Lab 18. Collect at least 100 episodes with fixed seeds. Validate event ordering and replay a sample. Include successful, failed, malformed, and truncated episodes.

**Submit:** dataset schema, replay tool, split strategy, and counts by failure type. **Checks:** no cross-episode state contamination, no tool observation mislabeled as an actor action, and no future result in an earlier observation.

<a id="chapter-40"></a>
### Chapter 40 — SFT for tool use

#### 40.1 Learn selection and arguments

Tool-use SFT must teach when to call a tool, which tool to choose, what arguments to send, and how to interpret its return. Syntactically valid JSON is only the first requirement.

Schemas define field types and constraints. A tool selection can be correct while its arguments are wrong. A model can also hallucinate a nonexistent tool or call a real one without sufficient information.

Multi-tool traces teach sequencing. Parallel calls are appropriate only when the calls' inputs and effects do not depend on each other. Training on parallelized traces that actually require ordering teaches a broken dependency graph.

#### 40.2 Error recovery and abstention

Include examples of missing records, timeouts, malformed arguments, and ambiguous requests. Teach retry only when the failure is plausibly transient; teach correction when arguments are wrong. Abstention can mean asking for missing information or stopping when the task cannot be completed under the available tools.

Assistant-generated tool-call tokens receive loss; environment-generated tool outputs usually do not. In multi-turn packing, preserve the visibility boundaries for each assistant action.

#### Lab 20 [A; optional B/C] — Agent behavior cloning

**Objective:** learn from successful trajectories.

1. Extract oracle demonstrations from Lab 19 with correct causal masks.
2. Train the tiny policy on action selection and arguments.
3. Evaluate complete episode success and conditional action accuracy separately.
4. Add demonstrations recovering from common learner errors.
5. Compare success-only cloning with cloning plus recovery at equal example count.
6. Test unknown keys and unavailable tools.

**Submit:** parsing, argument, recovery, and episode metrics. **Check:** each training action is conditioned only on prior observations. **B/C extension:** use a small LM with the same tool schemas.

<a id="chapter-41"></a>
### Chapter 41 — RL for agents

#### 41.1 Optimize consequences of decisions

For an episode $\tau$ containing model actions $a_k$ and observations, the on-policy gradient is

$$
\nabla J=
\mathbb E_\tau\left[
\sum_k \nabla\log\pi_\theta(a_k\mid h_k)(G_k-b(h_k))
\right],
$$

for the undiscounted finite-horizon setting used here. If action $a_k$ is a token sequence, its log probability is the sum over its actor-generated tokens. Tool-result tokens are not policy actions.

Terminal success gives a sparse signal. Dense environment rewards or learned rewards can provide more feedback but may change what is optimized. SFT initializes a useful policy; RL then learns under states induced by its own actions. Mixing an SFT loss with RL can preserve demonstrated behavior, at the cost of a changed combined objective.

#### 41.2 Online rollouts and offline trajectories

Fresh on-policy rollouts match the score-function estimator. Offline traces from older policies need an explicit treatment: supervised imitation, off-policy correction, or an offline-RL method with its assumptions. Calling every update on stored successful traces “RL” obscures the distinction.

For long trajectories, full importance weights can explode or vanish. Local clipped surrogates improve practicality but should not be described as exact unbiased correction of the complete trajectory distribution.

#### 41.3 A concrete two-decision agent gradient

An agent first chooses whether to look up a missing value, with probability $p=\sigma(u)$. If it looks up the value, it submits the correct answer with probability $q=\sigma(v)$. Assume no success without lookup and reward 1 only for a correct submission.

Expected success is $J=pq$. Therefore

$$
\frac{\partial J}{\partial u}=q\,p(1-p),
\qquad
\frac{\partial J}{\partial v}=p\,q(1-q).
$$

At $p=0.5,q=0.8$, the derivatives are 0.2 and 0.08. Improving the submission policy increases the value of learning to look up, and improving lookup frequency gives the submission policy more training opportunities.

This simple coupling explains why agent curricula can help. If either $p$ or $q$ is nearly zero, terminal successes are rare. SFT can seed one decision so RL obtains useful signal for the other.

**Lab extension:** implement the model with two logits and compare sampled trajectory gradients with these exact derivatives before introducing text or tools.

#### Lab 21 [A; optional B/C] — Agent RL

**Objective:** optimize terminal task success.

Use the cloned agent from Lab 20. Begin with two-step tasks and a finite action policy. Train with terminal reward and an independent baseline or RLOO over episodes. Increase the horizon only after measuring stable short-horizon learning.

**Submit:** success by horizon, reward sparsity, tool-call count, and comparison with the frozen SFT agent. **Checks:** budgets match across policies, resets are isolated, and every actor log probability corresponds to an action it actually sampled. **B/C extension:** train a textual tool-use policy.

<a id="chapter-42"></a>
### Chapter 42 — Credit assignment for long-horizon agents

#### 42.1 Three levels of credit

Trajectory credit gives an entire episode one result. Step credit estimates the contribution of each decision. Token credit distributes learning within an action's generated text. The levels answer different questions.

A failed final answer may be caused by a bad search query ten actions earlier. Giving every token the same terminal reward is a valid coarse score-function construction, but its variance can be large and it does not explain the cause of failure.

Value functions predict expected future return. Advantages compare an action to the expected outcome at its state. Process rewards can indicate intermediate validity, but they are additional supervision, not automatically causal attribution.

#### 42.2 Counterfactual and hierarchical credit

Counterfactual analysis asks what would happen if one decision changed while other conditions were controlled. In a resettable simulator, rerun alternative actions from a stored state. In a live environment, such comparisons may be noisy or impossible.

A hierarchical policy chooses a subgoal and then lower-level actions. Credit can be assigned at both levels, but a subgoal's duration and termination rule matter. Discounting over variable-duration actions requires accounting for elapsed steps.

**Worked example.** An agent searches the wrong key, retrieves an irrelevant row, and calculates correctly on that row. A terminal failure does not mean the arithmetic action was locally wrong. A counterfactual replay replacing only the lookup key can identify the earlier decision as decisive.

> **Figure placeholder F06 — Agent credit across a trajectory.** Show a six-step task with an early misleading lookup and a later failure. Overlay terminal returns, value estimates, and one counterfactual branch. Caption: “A late outcome can depend on an early information-gathering action.”

**Exercise.** Explain why rewarding every retrieved document can produce unnecessary searches. Lab 22 returns to this after potential-based shaping in Chapter 48.

<a id="chapter-43"></a>
### Chapter 43 — Harnessed agentic RL

#### 43.1 Runtime and trainer have different jobs

The runtime executes tool interactions. The trainer consumes a representation of those interactions and updates weights. Decoupling them allows the same harness to use different trainers, but creates a contract that must be exact.

An API gateway can capture each model call and response, recording prompt tokens, generated tokens, sampling parameters, probabilities, and version IDs. It must distinguish actual policy decisions from deterministic formatting inserted by the harness.

Training sample construction maps each decision boundary to a context, target actions, reward/advantage, and mask. A single episode can yield several training examples.

#### 43.2 Retokenization and merging pitfalls

Retokenizing stored text with a changed tokenizer can change token boundaries. Recorded old log probabilities then refer to different events and cannot be reused blindly. Preserve original token IDs for probability-based updates.

Sample merging can reduce overhead by combining decisions into one sequence. Ensure tool outputs remain unsupervised and repeated prefix tokens do not receive duplicated actor gradients. Context truncation during merging changes what the trainer sees relative to what the actor saw.

Weight synchronization must attach a version to each action, especially when an episode spans updates. Attribution asks which policy and harness components produced the success, not merely which final checkpoint existed when the episode ended.

**Case study reading:** [Agent Lightning][lightning] discusses separation between agent execution and training. Use its architecture as a case to reconstruct, not as a requirement for the labs.

**Exercise.** Design a validator that rejects a training sample if the token IDs, policy version, behavior probabilities, or model-visible history are missing.

<a id="chapter-44"></a>
### Chapter 44 — Training web and search agents

#### 44.1 Information gathering is a policy

A search agent chooses queries, pages, and stopping decisions. Query formulation controls recall and precision; opening a result spends budget and changes the evidence available. Browsing also involves navigation and page-state changes.

Reward should assess the final claim and its evidence. Counting citations encourages citation quantity. A useful citation must support the claim, come from the cited page, and refer to evidence available during the episode.

#### 44.2 Controlled environments and curricula

Start with a local frozen document collection. It makes resets, hidden answers, and reproducibility manageable. Later add distractors, contradictory sources, missing information, and query ambiguity.

Live pages change, so a replay requires snapshots or an acknowledged time-dependent environment. Reset state such as cookies, tabs, accounts, and partially completed forms. [WebArena][webarena] is a primary example of a benchmark built around interactive web environments.

**Worked example.** An agent answers a question correctly from memorized knowledge but cites an unrelated page. Outcome accuracy alone passes it; evidence-grounding evaluation fails it.

**Exercise.** Define separate metrics for answer correctness, source support, query cost, and successful abstention when evidence is absent.

<a id="chapter-45"></a>
### Chapter 45 — Training coding agents

#### 45.1 Repository state is part of the environment

A coding task begins with a repository snapshot and an issue or specification. Actions inspect files, execute commands, edit code, and run tests. The final artifact is often a patch.

Tests provide executable feedback, but test coverage determines what is checked. Patch rewards may include correctness, minimality, compatibility, and style. These can conflict: a tiny patch can be wrong, while a larger refactor can be correct but risky.

Long trajectories include exploration, failed edits, diagnostics, and recovery. Preserve them when analyzing credit assignment. A successful final patch does not prove every earlier action was useful.

#### 45.2 Isolation and contamination

Use isolated task directories and bounded processes. Keep evaluation tests outside the agent-editable surface. A modified test suite can make a bad patch appear successful.

Contamination includes training on the task's accepted patch or near-identical issue. Split by repository or time where feasible, and report the limitations. [SWE-bench][swebench] provides a primary task formulation based on resolving repository issues.

#### Lab 23 [A] — Coding-agent environment

**Objective:** give the agent tiny repositories and tests.

Create five small packages with seeded bugs: an off-by-one loop, a bad boundary condition, a sorting error, an incorrect default, and a state-reset bug. Provide read-file, apply-patch, and run-public-tests actions. Hold back an independent test suite.

**Submit:** resettable tasks, oracle patches, action logs, and public/hidden test outcomes. **Checks:** each episode starts from a clean snapshot, hidden tests cannot be edited, and timeouts are represented distinctly from incorrect outputs. Use a finite set of candidate edits for the first CPU policy.

<a id="chapter-46"></a>
### Chapter 46 — Computer-use agents

#### 46.1 Perception and action grounding

Computer-use policies receive screens or accessibility information and issue clicks, key presses, or text entry. Coordinates must be grounded in a particular viewport. Resizing, scrolling, and delayed loading can make the same coordinate refer to different controls.

A vision-language policy must connect instructions with visual entities. Two screens can look alike while differing in hidden state, creating partial observability. Long horizons amplify small perception and action errors.

#### 46.2 Reward and robustness

Define success through application state where possible, rather than a screenshot that merely looks complete. UI changes, popups, and latency require recovery behavior.

Actions can have irreversible effects. Training environments should make consequences explicit and support safe resets. Capability includes recognizing when available observations are insufficient for a consequential action.

**Worked example.** A model sees a “Save” button at the same location before and after switching tabs. Without retaining which form is active, a visually plausible click may save the wrong object.

**Exercise.** Build a synthetic two-screen GUI task and vary layout while preserving semantics. Measure whether the policy learned coordinates or grounded control selection.

<a id="chapter-47"></a>
### Chapter 47 — Multi-agent post-training

#### 47.1 Joint behavior and nonstationarity

In cooperative settings, agents share a return. In competitive settings, rewards conflict. Communication becomes an action with bandwidth and cost. Training one agent changes the environment faced by another, creating nonstationarity.

Centralized training can use joint state or actions in a critic while decentralized execution restricts each actor to its own information. This information boundary must be respected at evaluation.

For agent $i$, a counterfactual baseline can average over its alternative actions while holding others fixed:

$$
b_i(s,a_{-i})=
\sum_{a'_i}\pi_i(a'_i\mid h_i)Q(s,a'_i,a_{-i}).
$$

Then $Q(s,a)-b_i(s,a_{-i})$ estimates the agent's relative contribution under that critic. Accuracy of the critic and the assumed factorization matter.

#### 47.2 Roles, debate, and self-play

Planner, actor, critic, and verifier roles can specialize. Debate can expose errors if evaluation rewards accurate arguments; it can also optimize persuasive but false statements. More voices do not create independent evidence automatically.

Self-play can improve against recent opponents while forgetting older strategies. Maintain an opponent population and evaluate cross-play, role swaps, communication failures, and held-out partners.

**Worked example.** Two agents choose A or B and succeed only if they match. A centralized solution may coordinate perfectly using shared information unavailable at deployment. Testing only training-time coordination overstates the deployed policy.

**Part XI deliverable:** a trained small agent, a replayable trajectory dataset, and a failure analysis that distinguishes policy, harness, tool, and environment errors.

<a id="part-xii"></a>
## Part XII — Reward Engineering and Verifiers

**Chapter navigation:** [48 · Designing good rewards](#chapter-48) · [49 · Verifiers](#chapter-49) · [50 · Reward hacking](#chapter-50)

**Learning goals:** reason about incentives before training, distinguish policy-preserving shaping from arbitrary extra rewards, and audit the checker itself.

<a id="chapter-48"></a>
### Chapter 48 — Designing good rewards

#### 48.1 Rewards define tradeoffs

A reward is a specification of what the optimizer is encouraged to do. Sparse rewards closely match an endpoint but can be hard to discover. Dense rewards provide frequent signal but can privilege intermediate behavior over actual completion.

Normalize scales deliberately. Combining a correctness score in $[0,1]$ with an unbounded judge score lets the latter dominate unless constrained. Reward decomposition makes it possible to detect which component improved.

A weighted sum chooses a tradeoff among objectives. Pareto-efficient policies are those for which improving one objective requires worsening another. A single weight vector generally reveals only part of the tradeoff surface.

#### 48.2 Potential-based shaping

For potential $\Phi(s)$, add

$$
F(s,a,s')=\gamma\Phi(s')-\Phi(s).
$$

The discounted shaping return telescopes:

$$
\sum_{t=0}^{T-1}\gamma^tF_t
=-\Phi(s_0)+\gamma^T\Phi(s_T).
$$

If the terminal contribution is zero or otherwise constant across policies, this adds a policy-independent quantity and preserves optimal policy ordering. In discounted infinite-horizon problems, bounded potentials make the terminal term vanish. In finite episodes, handle terminal potentials explicitly. Arbitrary “progress bonuses” do not inherit this property. [Policy invariance under reward transformations][shaping]

**Example.** Let potential be negative remaining distance to a goal, with terminal potential zero and $\gamma=1$. Moving closer produces a positive shaping reward, moving away a negative one, and the total shaping contribution depends only on the start and terminal conventions.

#### 48.3 A shaping counterexample

Consider an episode starting at potential 0, with two terminal outcomes. Both have true reward zero, but terminal potentials are 1 and 0. With $\gamma=1$, adding potential differences gives total shaped reward 1 for the first outcome and 0 for the second.

The policy ranking changed because the terminal term was not constant. This does not contradict the invariance result; it violates its boundary condition.

Similarly, giving +1 every time an agent retrieves a relevant document is not automatically potential-based. If the agent can retrieve the same document repeatedly, it may accumulate reward without completing the task. A potential based on newly acquired information can avoid some loops, but its terminal handling and definition still require analysis.

**Exercise:** create a loop in the toy environment. Compare arbitrary per-action bonuses with a bounded potential difference. Verify that traversing a closed loop returns net shaping reward zero when $\gamma=1$.

#### Lab 22 [A; optional B/C] — Credit assignment study

**Objective:** compare terminal rewards with shaping and process rewards.

Use the same multi-step environment, policy initialization, task distribution, and interaction budget for three conditions: terminal reward, valid potential-based shaping, and a learned process reward. Add an intentionally exploitable “reward every lookup” condition.

**Submit:** success, episode length, reward components, and sample efficiency across seeds. **Checks:** demonstrate the telescoping identity numerically; do not claim learned process rewards preserve optimality. **Research question:** does a denser signal improve true success or merely the proxy?

<a id="chapter-49"></a>
### Chapter 49 — Verifiers

#### 49.1 A hierarchy of evidence

Rules check explicit conditions. Unit tests check selected behaviors. Symbolic tools reason within formal assumptions. LLM judges estimate judgments. Each has a distinct failure model.

| Verifier | Strength | Typical blind spot |
|---|---|---|
| Exact schema/rule | Deterministic and inspectable | Underspecified semantics |
| Unit tests | Executable behavior | Incomplete input coverage |
| Symbolic/proof checker | Strong within formal specification | Wrong specification or assumptions |
| Model grader | Flexible qualitative assessment | Bias, inconsistency, exploitation |
| Ensemble | Multiple signals | Correlated errors |

Reference-based grading compares to an answer or evidence. Reference-free grading judges intrinsic quality. Pairwise grading compares alternatives and can be more stable for some tasks, but it cannot directly certify absolute acceptability.

#### 49.2 Calibration and robustness

If a grader outputs a probability, evaluate calibration against independent labels. Brier score is the mean of $(p_i-y_i)^2$; it measures probabilistic accuracy, while reliability diagrams reveal systematic overconfidence.

Use adversarial examples, perturbations, and distribution shifts. A robust math checker should handle equivalent notation without accepting unrelated strings. A code verifier should fail invalid solutions even if they match public examples.

Do not treat a parser crash as proof of an incorrect solution. It is a distinct operational outcome, though the training reward may assign it zero.

#### Lab 24 [A; optional B/C] — Test-based coding-agent RL

**Objective:** use test results as the verifier.

1. Reuse the immutable task snapshots from Lab 23.
2. Let a finite CPU policy choose among candidate patches, then extend to sequential edits.
3. Reward public-test outcomes during training.
4. Evaluate hidden tests only after selecting checkpoints with a separate validation procedure.
5. Add patches that overfit public examples and inspect the gap.
6. Track invalid patches, timeouts, partial passes, and full hidden-test success.

**Submit:** reward contract, patch examples, learning curves, and public/hidden generalization. **Check:** the agent cannot change the training reward implementation or held-out tests. **B/C extension:** replace candidate-patch selection with a language policy that writes edits.

<a id="chapter-50"></a>
### Chapter 50 — Reward hacking

#### 50.1 Optimize the intended task or the measurement?

Specification gaming exploits a mismatch between what is measured and what is wanted. Examples include adding keywords favored by a grader, printing expected test output without implementing the function, using excessive formatting to exploit a parser, or modifying the environment so success is falsely reported.

Distributional hacking need not involve an explicit exploit: optimization can concentrate on inputs where the grader is predictably wrong. Memorizing a verifier's known test cases is another route.

#### 50.2 Detection and repair

Use an independent audit signal, inspect extreme-reward samples, and compare reward with true outcomes. Perturb irrelevant formatting and check stability. Preserve a held-out set of verifier attacks.

Repair the specification, parser, or evaluation boundary that failed. Retraining the policy on a few attacks may help, but it does not repair an incomplete test suite by itself.

**Worked example.** A grader rewards any response containing “42.” A policy learns to append “42” to every answer. Fixing this means defining the answer field and comparing it to the task-specific result, not simply banning one token.

#### Lab 16 [A] — Reward hacking

**Objective:** introduce a flawed verifier and document the exploit.

In the local arithmetic environment, compare a loose substring checker with Lab 14's strict checker. Optimize the same finite policy against each. Use a separate true-answer evaluator for both.

**Submit:** the exact specification gap, discovered high-reward failures, proxy/true-accuracy curves, and results after repair. **Check:** all experimentation stays inside the synthetic environment. **Research extension:** test whether repairing one loophole creates pressure toward another.

<a id="part-xiii"></a>
## Part XIII — Data Engineering for Post-Training

**Chapter navigation:** [51 · Prompt distributions](#chapter-51) · [52 · Synthetic data generation](#chapter-52) · [53 · Data flywheels](#chapter-53)

**Learning goals:** define the training distribution and make each data transformation traceable.

<a id="chapter-51"></a>
### Chapter 51 — Prompt distributions

#### 51.1 The prompt distribution is part of the objective

The objective averages over $x\sim D$. Changing $D$ changes what is optimized, even when the response-level algorithm is unchanged.

Let deployment mixture be 60% retrieval, 30% reasoning, and 10% tool control. Training equally on the three tasks intentionally overweights tool control. Report per-domain metrics and performance under the deployment mixture.

Difficulty and long-tail behavior matter. Average performance can improve while rare critical tasks regress. Stratified sampling gives small domains enough observations to estimate their behavior.

#### 51.2 Weighting and adaptive curricula

If training prompts come from $q(x)$ but the target is $D(x)$, importance weights $D(x)/q(x)$ can correct expectations when distributions and support are known. Large weights create variance. In practice, deployment distributions are often estimated imperfectly.

Hard-example mining emphasizes failures. A dynamic curriculum changes as the policy improves. Log how prompts were selected and preserve an evaluation distribution independent of that curriculum.

**Example.** An algorithm discards all-success and all-failure groups. Its accepted prompt distribution depends on current success probabilities. The trainer is no longer averaging over the original prompts uniformly.

**Exercise.** Specify a sampler that balances domain coverage with informative difficulty. List the metrics needed to detect neglect of very easy and very hard tasks.

<a id="chapter-52"></a>
### Chapter 52 — Synthetic data generation

#### 52.1 Generate the missing kind of evidence

Synthetic generation can produce prompts, responses, preferences, critiques, and whole interactive tasks. Choose the object that addresses the current limitation. More paraphrases of solved tasks may not help a policy that fails new reasoning structures.

Teacher identity, sampling settings, prompts, and verification must be versioned. Use executable task generators when ground truth can be produced independently. A teacher-generated answer and a teacher-generated correctness label can share the same error.

#### 52.2 Diversity and filtering

Control diversity through task families, difficulty, surface forms, and solution strategies. Check diversity after filtering, because a strict filter can erase most of what was generated.

Synthetic-data collapse refers to loss or distortion of distributional support through repeated generation and retraining under particular conditions. It is not a theorem that every use of synthetic data fails. Retaining diverse reliable sources and independent verification changes the process.

**Worked example.** Generate equation-solving problems by first choosing a solution and constructing an equation. This gives independent ground truth. Asking a teacher to invent an equation and its answer provides less independence.

**Exercise.** Design a generation pipeline with separate task creator, solver, checker, and diversity audit. State which failures remain correlated.

<a id="chapter-53"></a>
### Chapter 53 — Data flywheels

#### 53.1 Turn failures into targeted data

A data flywheel collects interactions, mines failure patterns, obtains corrections or grader labels, retrains, and re-evaluates. Its benefit depends on whether the new labels address real deployment errors.

Failure mining should classify root causes: missing knowledge, bad retrieval, invalid tool arguments, poor reasoning, or harness failure. Not every problem is best fixed by updating model weights.

Human correction can provide a desired response or an explanation of what was wrong. Preserve that distinction. A grader's score is weaker evidence than an executable outcome when the latter is available.

#### 53.2 Avoid feedback-loop illusions

Users who remain active may not represent those who abandoned the product. A policy can change which users or tasks generate feedback. Online learning can then reinforce a selection bias.

Use stable evaluation sets plus newly collected representative samples. Keep training, validation, and audit data boundaries explicit. Compare releases before feeding their evaluation outcomes into the next training set.

**Exercise.** A retrained agent shows fewer reported errors but lower task completion. Explain how reduced usage, premature stopping, or discouraged feedback could produce this pattern.

<a id="part-xiv"></a>
## Part XIV — Evaluation Science

**Chapter navigation:** [54 · Offline evaluation](#chapter-54) · [55 · LLM-as-a-judge](#chapter-55) · [56 · Agent evaluation](#chapter-56) · [57 · Benchmark pathology](#chapter-57) · [58 · Experimental design](#chapter-58)

**Learning goals:** define the estimand, quantify uncertainty, and compare methods without changing several important variables at once.

<a id="chapter-54"></a>
### Chapter 54 — Offline evaluation

#### 54.1 Choose the quantity you intend to estimate

Accuracy measures the fraction of correct tasks under a defined sampling and decoding procedure. Exact match is one operationalization, with known equivalence limitations. Mean reward measures the chosen proxy. Win rate compares against a specified opponent and judge.

Pass@k estimates the probability that at least one of $k$ generated samples is correct. If a task has $n$ sampled candidates with $c$ successes and $n\geq k$, the familiar estimator is

$$
\widehat{\text{pass@k}}
=1-\frac{\binom{n-c}{k}}{\binom nk}.
$$

It comes from the probability that a uniformly selected subset of $k$ candidates contains no successes. Under the corresponding independent sampling setup it estimates the desired multi-sample success probability. Correlated decoding or adaptive candidate selection changes the interpretation. [Evaluating Large Language Models Trained on Code][humaneval]

Pass@k assumes access to a correctness oracle for evaluating whether any sample succeeded. A deployed selector may fail to identify that successful sample.

#### 54.2 Uncertainty and calibration

For a binary proportion $\hat p$, a rough standard error is $\sqrt{\hat p(1-\hat p)/n}$. Near 0 or 1 or at small $n$, use a more suitable interval such as Wilson:

$$
\frac{\hat p+z^2/(2n)\ \pm\
z\sqrt{\hat p(1-\hat p)/n+z^2/(4n^2)}}
{1+z^2/n}.
$$

For 95% nominal coverage, $z\approx1.96$. The interval reflects sampling assumptions, not benchmark representativeness.

Bootstrap tasks to estimate uncertainty in an aggregate metric. For paired model comparisons, resample the same task indices for both. For training randomness, report multiple seeds; one seed evaluated on many tasks does not measure variability across training runs.

Calibration asks whether confidence matches empirical correctness. Separate the model's self-reported confidence from a calibrated probability estimator.

**Exercise.** Compute pass@2 for $n=5,c=2$: $1-\binom32/\binom52=0.7$. Explain why this does not equal the observed pass@1 of 0.4.

#### 54.3 Separate task and training uncertainty

Suppose one trained checkpoint beats another on 1,000 tasks. A paired bootstrap over those tasks estimates uncertainty conditional on those two checkpoints. It does not answer how often the training procedure would produce a better checkpoint if rerun.

To study the procedure, train multiple seeds and retain per-task results for each. You can report the distribution of seed-level scores and use a hierarchical analysis that respects both sources of variation. Avoid treating every seed-task pair as independent if tasks or training conditions are shared.

Another issue is checkpoint selection. If you evaluate 100 checkpoints and report the best test score, the chosen score is optimistic even when each individual estimate is unbiased. Select with validation data, then use a final test once for the declared comparison.

**Exercise:** simulate two identical methods with noisy scores and select the best of 1, 10, and 100 runs. Observe how “best score” improves without any method improvement.

<a id="chapter-55"></a>
### Chapter 55 — LLM-as-a-judge

#### 55.1 Make the rubric inspectable

A judge prompt should define the task, criteria, evidence, response format, and tie handling. Pairwise judging compares responses; reference-based judging can use a known answer or supporting material. Neither eliminates errors.

Position bias favors one slot. Verbosity bias favors more text. Self-preference can favor outputs resembling the judge's own style or model family. Test these by controlled swaps and perturbations.

The judge should receive response content as data. Instructions embedded in candidate responses can attack the grading process. Test robustness to those instructions.

#### 55.2 Validate against independent judgments

Compare model judgments with human labels on a representative stratified subset. Report disagreements by error type, not just aggregate agreement. Calibrate scores if they are used as probabilities or thresholds.

An ensemble can reduce some noise. If all judges share a similar bias or prompt, their votes are not independent evidence. Audit adversarial and boundary cases separately. [MT-Bench and Chatbot Arena judging study][judge]

**Worked example.** Evaluate each pair in both orders. If A wins only when shown first, the result is unstable. A balanced aggregate can reduce position effects, but does not prove correctness.

**Exercise.** Design a judge audit with equal-content length changes, swapped order, intentionally false but fluent answers, and answers containing grader-directed instructions.

<a id="chapter-56"></a>
### Chapter 56 — Agent evaluation

#### 56.1 Success includes the whole interaction

Measure task completion, tool calls, steps, latency, monetary or compute cost, recovery, and robustness. A policy that succeeds more often by using ten times the budget has a different operating point, not an unqualified improvement.

Trajectory quality includes unnecessary actions, repeated errors, unsupported claims, and whether the environment reached the intended state. A final answer can sound correct while the requested change never happened.

#### 56.2 Environmental variance

Agents can face nondeterministic search results, UI timing, tool failures, or state resets. Evaluate multiple environment seeds and preserve snapshots when possible.

Use success-versus-budget curves. Compare expected cost and tail latency. Two policies with identical average calls can have very different rates of getting stuck in loops.

**Example.** Agent A succeeds on 80% of tasks with five calls; B succeeds on 85% with twenty calls. Plot both across budgets before deciding which is better for the application.

**Exercise.** Create a failure taxonomy distinguishing perception, planning, tool selection, arguments, execution, recovery, and evaluation errors. Label ten trajectories independently of their training rewards.

<a id="chapter-57"></a>
### Chapter 57 — Benchmark pathology

#### 57.1 A benchmark is a measurement instrument

Contamination exposes the learner to evaluation answers or close equivalents. Repeated public evaluation can cause human-in-the-loop overfitting even when the test data is never directly included in training.

Saturation reduces the ability to distinguish methods. Gaming exploits a metric. Distribution shift tests whether improvement transfers beyond the benchmark's narrow setting. Validity asks whether the benchmark measures the capability you actually claim.

Hidden tests reduce direct adaptation, but their coverage and quality still matter. Private evaluation suites can cover product-specific behavior; they should have documented construction and independent review.

#### 57.2 Diagnose suspicious gains

Compare results on newly generated tasks, changed surface forms, and held-out domains. Inspect nearest training examples and unusual answer patterns. A clean split at the row level can still leak through shared templates or repositories.

**Worked example.** A coding model memorizes an accepted patch but cannot fix the same bug after variables are renamed. The original success did not demonstrate the intended generalization.

**Exercise.** Write three distinct claims a benchmark could support and three it could not. For example, performance on a small deterministic coding suite does not establish reliability on arbitrary production repositories.

<a id="chapter-58"></a>
### Chapter 58 — Experimental design

#### 58.1 Define an estimand and a fair comparison

An estimand is the quantity being estimated: mean held-out accuracy difference at a fixed rollout budget, for example. Define it before choosing metrics.

A baseline should be competent and receive an appropriate tuning budget. Compare with SFT, an unchanged policy, and a simple optimizer when relevant. An ablation removes or changes one proposed mechanism. Factorial designs reveal interactions between mechanisms.

Match model initialization, prompts, reward, sampling, evaluation, and data budget when isolating an optimizer. Compute matching requires counting discarded rollouts, judge calls, failed environments, and hyperparameter sweeps. Equal update counts alone are rarely equal compute.

#### 58.2 Seeds, curves, and statistical evidence

Use a pilot to estimate variability and choose sample size for an effect worth detecting. More seeds measure training variability; more evaluation tasks measure task variability. Neither substitutes for the other.

Show learning curves against updates, generated tokens, and wall time where relevant. Predefine stopping and checkpoint selection. Selecting the best test checkpoint introduces optimistic bias.

For paired task outcomes $d_i=a_i-b_i$, estimate $\bar d$ and uncertainty on those differences. Report practical effect size with confidence intervals. A small p-value is not a measure of usefulness or replication probability.

> **Figure placeholder F07 — A defensible algorithm comparison.** Show individual-seed learning curves and uncertainty bands against total generated tokens. Add an inset with the fixed data, model, and reward controls.

#### 58.3 A complete experiment specification

For a comparison of group normalization, write the following before collecting final results:

| Item | Example specification |
|---|---|
| Question | Does per-prompt standardization change learning on mixed-difficulty tasks? |
| Primary metric | Held-out mean success under uniform task sampling |
| Intervention | Standardization on/off |
| Fixed quantities | Initial logits, reward values, prompt set, rollout count |
| Tuning | Same declared learning-rate candidates and validation budget |
| Repetitions | Independent training seeds chosen before execution |
| Selection | Best validation checkpoint within a fixed interaction budget |
| Analysis | Paired task differences and seed-level variability |
| Secondary metrics | Entropy, length, gradient variance, group acceptance |
| Failure rule | Report all numerical failures and rerun only documented implementation errors |

The specification reduces freedom to explain away an unfavorable outcome. It also clarifies what a negative result means: no detected effect under these tasks, budgets, and sensitivity, rather than proof that the component can never matter.

#### Lab 13 [A; optional B/C] — Algorithm comparison

**Objective:** hold model, prompts, and rewards fixed across GRPO, RLOO, and REINFORCE.

1. Predefine a held-out task distribution and primary metric.
2. Use the same initial policy, reward, and total rollout budget.
3. Give each method a declared, comparable hyperparameter search budget.
4. Log all generated responses, including filtered groups.
5. Run multiple seeds and plot performance against interactions and runtime.
6. Add one ablation of normalization while holding sampling fixed.
7. Report uncertainty, failures, and the exact scope of the conclusion.

**Submit:** reproducible configuration, all run summaries, paired evaluation, and a one-page conclusion. **Check:** do not claim a method is universally superior from a single toy environment. **B/C extension:** test whether the observed mechanism persists on a small LM.

**Part XIV mastery check:** explain exactly what your comparison isolates, what remains confounded, and how large an effect the experiment could reliably detect.

<a id="part-xv"></a>
## Part XV — Post-Training Systems Engineering

**Chapter navigation:** [59 · Rollout infrastructure](#chapter-59) · [60 · Training/inference disaggregation](#chapter-60) · [61 · Distributed post-training](#chapter-61) · [62 · RL-specific systems problems](#chapter-62)

**Learning goals:** estimate resource use, separate training and rollout responsibilities, and diagnose systems effects that alter the learning algorithm.

<a id="chapter-59"></a>
### Chapter 59 — Rollout infrastructure

#### 59.1 Generation is a major training workload

An RL trainer repeatedly generates data with its evolving policy. Rollout workers perform inference; training workers compute gradients. The optimal hardware arrangement for one is not automatically optimal for the other.

Generation consists of prefill, which processes the prompt, and decode, which emits tokens sequentially. Batching improves utilization, but responses finish at different times. Continuous batching admits new work as capacity becomes available rather than waiting for a whole fixed batch to finish.

Serving engines manage model execution, batching, memory, and scheduling. PagedAttention is one specific memory-management approach described in the [vLLM paper][pagedattention]. Treat that paper as a mechanism study; production APIs evolve independently.

#### 59.2 KV-cache accounting

For a decoder with $L$ layers, batch $B$, cached length $T$, $n_{\mathrm{kv}}$ key/value heads, head dimension $d_h$, and $b$ bytes per element, a simple cache estimate is

$$
M_{\mathrm{KV}}\approx2LBTn_{\mathrm{kv}}d_hb.
$$

The factor 2 counts keys and values. Grouped-query attention reduces the number of KV heads relative to query heads. Metadata, allocator overhead, page fragmentation, and temporary buffers are additional.

For $L=24,B=4,T=4096,n_{\mathrm{kv}}=8,d_h=64,b=2$, the estimate is 805,306,368 bytes, or 0.75 GiB. Doubling cached length doubles this estimate. This explains why a model whose weights fit can still run out of memory during long rollouts.

#### 59.3 Scheduling changes data collection

Short jobs complete sooner. If training consumes the first available results, it can overweight short or easy tasks. Scheduling is therefore potentially a sampling intervention.

Track prompt length, generated length, prefill/decode throughput, queue wait, tool wait, and rejected work. Agent environments introduce CPU and I/O bottlenecks beyond model generation.

**Exercise.** A rollout queue contains equal numbers of 10-token and 1,000-token responses. If each update uses the first 100 completed jobs, predict the initial training mixture and propose a sampling-preserving collection rule.

<a id="chapter-60"></a>
### Chapter 60 — Training/inference disaggregation

#### 60.1 Worker roles and topology

Actor trainers update the policy. Rollout workers sample it. Reward workers grade responses. Critic trainers update values. Reference scoring may have its own workers or be colocated. Disaggregation gives these stages different resources; colocation saves transfer or hardware costs but can create contention.

Weight synchronization broadcasts a checkpoint or parameter update to rollout workers. The receiver should acknowledge an exact version before serving new trajectories. An episode sampled across multiple versions requires per-decision attribution or an enforced snapshot policy.

#### 60.2 Synchronous and asynchronous loops

A synchronous loop collects a batch, trains, synchronizes, and repeats. It is simple but can idle workers while waiting for slow tasks. An asynchronous loop overlaps generation and training, improving utilization while introducing stale behavior data.

Let rollout service produce $r$ accepted samples per second and training consume $u$. If $r>u$, the queue grows unless bounded. If $r<u$, training starves. Little's law relates average in-flight work $L_q$, arrival rate $\lambda$, and mean time in system $W$: $L_q=\lambda W$ under a stable steady state. It is useful for capacity planning, not a guarantee under transient overload.

Policy age can be measured in versions, updates, elapsed time, or distributional KL. Version difference is easy to log but only a proxy for distribution mismatch.

**Case study:** [HybridFlow][hybridflow] illustrates coordination of distributed RLHF computation. Reconstruct the roles and transfers rather than memorizing framework components.

> **Figure placeholder F08 — Disaggregated training.** Show rollout, reward, trajectory-store, trainer, and evaluator services. Label data flow and weight-version flow separately. Mark queue capacity and stale-sample rejection.

**Exercise.** Design a rule for maximum policy age and a backpressure strategy. Explain what happens to expensive episodes already in progress when new weights arrive.

#### 60.3 A queueing example with statistical consequences

Assume rollout workers produce 20 completed trajectories per second, but the trainer consumes 12. The queue grows at eight trajectories per second during sustained operation. After a minute, roughly 480 extra trajectories are waiting, ignoring startup and variability.

A larger buffer prevents immediate blocking but increases policy age. Discarding the oldest samples bounds age while wasting generation. Dropping the longest tasks can improve throughput while biasing the training distribution. Backpressure slows producers and preserves accounting but lowers instantaneous utilization.

There is no universally best choice. Measure learning per total generated token and per wall-clock second, including discarded work. Track the task types that are delayed or dropped.

**D-tier experiment:** simulate two task-duration distributions, one narrow and one heavy-tailed, under the same mean duration. Compare synchronous batching and asynchronous collection. Report both throughput and the accepted-task mixture.

<a id="chapter-61"></a>
### Chapter 61 — Distributed post-training

#### 61.1 Match the parallelism to the bottleneck

Data parallelism replicates computation and combines gradients. Fully sharded methods reduce replicated state. Tensor parallelism partitions within-layer operations. Pipeline parallelism partitions layers. Sequence/context parallelism distributes long-context work. Expert parallelism routes tokens to different MoE experts.

Combining them introduces communication patterns that may conflict with rollout execution. A tensor-parallel arrangement ideal for inference may differ from one ideal for training.

MoE activates a subset of parameters per token, but total expert storage and routing communication still matter. Load imbalance can overload popular experts. Active parameter count is not total memory cost.

#### 61.2 Communication and overlap

An all-reduce combines gradients; an all-gather reconstructs sharded values; reduce-scatter combines and redistributes them. The exact bytes transferred depend on the algorithm, topology, and data type.

Estimate whether compute can overlap communication. A nominally faster kernel may expose communication as the new bottleneck. Small microbatches can lower memory while reducing utilization.

**Worked example.** If one update takes 4 seconds of compute and 3 seconds of nonoverlapped communication, doubling compute speed reduces total time from 7 to 5 seconds, not 3.5. Optimization must target the actual critical path.

#### 61.3 Checkpoint and recovery semantics

A resumable RL checkpoint needs more than actor weights: optimizer state, scheduler, critic, reference identity, random-number states, dataset/sampler state, reward version, and enough queue metadata to account for in-flight samples.

Fault recovery must prevent duplicate updates and mismatched versions. Exactly-once processing is difficult; idempotent sample IDs and explicit consumed-batch records make retries inspectable.

**Exercise.** Simulate a crash after gradients are applied but before a batch is marked consumed. Specify how your recovery protocol avoids silently applying the same update twice.

<a id="chapter-62"></a>
### Chapter 62 — RL-specific systems problems

#### 62.1 Learning and utilization can conflict

The highest hardware utilization may produce excessively stale samples. The freshest data may require idle periods. Tune the system against learning progress per wall time and resource budget, not utilization alone.

Stragglers arise from long responses, slow tools, or heterogeneous workers. Dropping them changes the data distribution. Padding shorter sequences wastes compute; packing them changes masking and reduction requirements.

Replay can improve reuse but increases off-policy mismatch. Log the behavior policy and sampling transform for every replayed decision. A generic replay buffer is not automatically appropriate for an on-policy objective.

#### 62.2 Numerical and representational mismatch

Different generation and training kernels can produce slightly different logits. Quantization and reduced precision can widen the discrepancy. Compare token log probabilities on identical inputs before training.

Tokenization drift is more severe: if token boundaries change, an old token probability is no longer the probability of the new token event. Retokenization requires an explicit reconstruction strategy, not shape-compatible tensor reuse.

Observability should connect sample IDs to prompt, behavior version, reward version, training batch, and checkpoint. Reproducibility includes both numerical variation and data-order variation.

#### Lab 25 [A + D; optional B/C] — Miniature post-training system

**Objective:** separate rollout server → verifier → trajectory store → trainer → evaluation server.

**A/D implementation:** use local Python processes or a deterministic event simulator. A finite policy and small queues are enough.

1. Define versioned request/response schemas and unique sample IDs.
2. Implement asynchronous rollout and verification with configurable delays.
3. Store accepted trajectories append-only.
4. Train on batches with a declared maximum policy age.
5. Publish versioned checkpoints and require worker acknowledgments.
6. Evaluate frozen checkpoints on separate tasks.
7. Inject a slow worker, duplicate delivery, verifier failure, and trainer restart.
8. Measure throughput, queue depth, age, sample loss, duplicate handling, and true learning.

**Submit:** architecture diagram, protocol, event log, recovery demonstration, and learning-versus-staleness comparison. **Checks:** no silent sample duplication, no mixed reward versions, and checkpoint recovery preserves the experiment's accounting. **B/C extension:** replace mock inference with a small model.

**Part XV mastery check:** explain how a scheduling or synchronization decision can alter the mathematical estimator, not just its runtime.

<a id="part-xvi"></a>
## Part XVI — Safety and Alignment Post-Training

**Chapter navigation:** [63 · Helpful, honest, and safe behavior](#chapter-63) · [64 · RLAIF and Constitutional AI](#chapter-64) · [65 · Adversarial post-training](#chapter-65) · [66 · Alignment tradeoffs](#chapter-66)

**Learning goals:** turn behavioral requirements into training and evaluation criteria, while making value conflicts and measurement limitations explicit.

<a id="chapter-63"></a>
### Chapter 63 — Helpful, honest, and safe behavior

#### 63.1 Behavioral requirements need examples and boundaries

Helpfulness includes completing the requested task and asking for missing information when needed. Honesty includes calibrated uncertainty and avoiding fabricated claims or evidence. Safety includes respecting constraints on consequential actions and harmful assistance.

These goals can conflict. A model rewarded for confidently satisfying every request may overstate knowledge. A model rewarded for avoiding all risk may refuse benign tasks. Measure both inappropriate compliance and over-refusal.

Sycophancy is behavior that follows a user's stated belief or desired answer despite contrary evidence. Evaluate paired prompts with the same facts but different user opinions. Truthfulness requires independent factual or executable checks, not just agreement with a judge.

#### 63.2 Instruction hierarchy and uncertainty

Training examples should distinguish trusted instructions from untrusted content such as retrieved pages or tool outputs. An agent must use external content as evidence without treating embedded instructions as authorized changes to its task.

Uncertainty calibration requires labels or outcomes. Repeatedly adding “maybe” is a style change, not proof of calibrated confidence. Evaluate whether uncertainty predicts error and whether the policy seeks useful information.

**Exercise.** Build a small matrix of benign versus disallowed tasks and clear versus ambiguous wording. Measure completion, justified refusal, unnecessary refusal, and clarification behavior separately.

<a id="chapter-64"></a>
### Chapter 64 — RLAIF and Constitutional AI

#### 64.1 AI feedback changes the source of supervision

RLAIF obtains preferences, critiques, or rewards from an AI evaluator. It can scale feedback, but inherits evaluator errors and preferences. A different feedback source does not eliminate the need to validate the reward.

Constitutional AI uses stated principles to guide critique and revision and to generate preference supervision in its described pipeline. A supervised phase can train revised responses; a preference/reinforcement phase can reinforce principle-consistent behavior. [Constitutional AI][constitutional]

#### 64.2 Principles must be operationalized

A principle such as “be honest” is broad. Examples, comparison prompts, and adjudication rules translate it into labels. Conflicting principles require an ordering or resolution procedure.

Scalable supervision asks how weaker or cheaper oversight can guide more capable systems. Decomposition, critique, and AI feedback are possible mechanisms, each with failure cases. An evaluator that cannot recognize a subtle error may reward a more persuasive wrong answer.

**Worked example.** A critic flags unsupported certainty, and a revised answer adds cautious wording while retaining the false claim. A style-only judge may accept it. A factual evaluator catches the unchanged error.

**Exercise.** Compare principle-guided feedback with a task-specific executable checker. Identify which behaviors each can evaluate and which remain outside its scope.

<a id="chapter-65"></a>
### Chapter 65 — Adversarial post-training

#### 65.1 Train on failures that expose a mechanism

Red teaming searches for failures under a stated threat model. Adversarial SFT demonstrates desired behavior on challenging inputs. Preference learning ranks robust responses above failed ones. Reward-model attacks test whether the evaluator can be manipulated.

Useful attack sets include paraphrases, role confusion, conflicting instructions, misleading tool results, and distribution shifts. Keep a held-out family of attacks to test generalization rather than memorization.

#### 65.2 Regression and agent risks

A safety improvement can reduce useful task completion or shift failures elsewhere. Evaluate benign boundary cases, ordinary tasks, and previously solved attacks after every change.

Agent risks include unauthorized tool use, unintended side effects, persistent state changes, and failure to stop when required. The harness can enforce boundaries in addition to training the policy. Measure the complete system.

**Exercise.** In the synthetic search environment, place an instruction inside a retrieved document that conflicts with the user's task. Train a policy to retain evidence while ignoring the unauthorized instruction. Evaluate on unseen wording and document layouts.

<a id="chapter-66"></a>
### Chapter 66 — Alignment tradeoffs

#### 66.1 One scalar does not resolve disagreement

Helpfulness, harmlessness, honesty, and capability are not a single universally agreed scale. Preference datasets may aggregate different groups and contexts. A reward model's output encodes that aggregation, including its compromises.

Pluralistic preferences can require conditioning on legitimate user preferences or offering alternatives. Personalization should not silently override requirements such as truthfulness or authorization.

Model specifications and constitutions provide behavioral targets. They need interpretable examples and regression suites. A specification is not a proof that a trained model follows it.

#### 66.2 Evaluate tradeoff curves

Vary reward weights or decision thresholds and plot useful-task completion against failure rates. Compare at matched operating points. A system with fewer harmful completions because it refuses almost everything has a different tradeoff from one that distinguishes cases accurately.

**Worked example.** Two policies have the same aggregate safety score, but one over-refuses medical vocabulary in benign educational tasks while the other fails tool authorization. Aggregate scoring hides different practical problems.

**Part XVI deliverable:** a behavioral evaluation matrix with clear criteria, boundary cases, and separate measures for compliance, refusal, uncertainty, and task success.

<a id="part-xvii"></a>
## Part XVII — Specialized and Frontier Topics

**Chapter navigation:** [67 · Multimodal post-training](#chapter-67) · [68 · Long-context post-training](#chapter-68) · [69 · Tool and retrieval specialization](#chapter-69) · [70 · Continual and online learning](#chapter-70) · [71 · Meta-learning and automatic curriculum generation](#chapter-71) · [72 · Open research problems](#chapter-72)

**Learning goals:** transfer the course's definitions to new modalities, long contexts, retrieval, and continual adaptation, while identifying unresolved assumptions.

<a id="chapter-67"></a>
### Chapter 67 — Multimodal post-training

#### 67.1 Condition on more than text

A vision-language policy conditions on text and an encoded image or sequence of frames:

$$
\pi_\theta(y\mid x,v).
$$

Visual instruction tuning uses demonstrations of responses grounded in visual input. Preference learning compares outputs under the same input. Multimodal reward may assess object grounding, spatial relations, OCR, factual correspondence, or action success. [Visual Instruction Tuning][llava]

The input representation matters: patch size, cropping, resolution, frame selection, and visual encoder can limit what the model observes. A reward cannot teach the policy to use visual information that preprocessing consistently removes.

#### 67.2 Grounding and multimodal reasoning

Grounding connects language or actions to specific visual evidence. A correct-looking answer can come from language priors. Test counterfactual images with the same question, changed object attributes, and distractors.

For computer use, vision-agent RL connects perceived controls to executable actions. For video agents, temporal ordering and persistent identity matter. Sampling isolated frames can miss the event that determines the answer.

**Worked example.** A model answers “red” for a familiar object even after the image is recolored blue. High accuracy on a biased dataset may reflect a language prior rather than visual grounding.

**Exercise.** Build a CPU synthetic image task with colored shapes and a small classifier/policy. Compare true visual input, shuffled images, and text-only input. This isolates the grounding mechanism without training a large multimodal model.

<a id="chapter-68"></a>
### Chapter 68 — Long-context post-training

#### 68.1 More context changes cost and learning

Standard dense attention has quadratic pairwise interaction cost in sequence length, while KV storage scales linearly with cached length during decoding. Efficient kernels reduce materialization and memory traffic; they do not automatically remove the underlying dense attention computation.

Context curricula gradually increase length or vary it during SFT. The model must learn to use distant relevant information, not just accept a longer tensor. Evaluate distractor robustness and position sensitivity.

Sequence/context parallelism distributes work across devices. Ring Attention is a primary example of blockwise attention computation and communication for long sequences. [Ring Attention][ring]

#### 68.2 Long rollouts and memory

Long-rollout RL increases the delay between early choices and final reward. It also changes truncation frequency, normalization, and total compute. A policy can exploit the budget or become dependent on extensive reasoning.

Memory compression reduces context cost but can lose decisive facts. Train and evaluate the compression policy separately when possible. A summary should preserve task-relevant uncertainty and constraints, not just fluent narrative.

**Worked example.** A relevant constraint at position 100 is omitted from a summary before a final action at position 20,000. The later error is partly a memory-policy failure. Training only the final answer head may not repair it.

**Exercise.** Vary evidence position and distractor length independently. Plot accuracy against both. Then compare full history with a fixed-budget summary and report what information is lost.

<a id="chapter-69"></a>
### Chapter 69 — Tool and retrieval specialization

#### 69.1 Retrieval can be an action policy

A retrieval agent chooses whether to search, how to rewrite a query, which source to use, and when enough evidence has been gathered. Reward can depend on downstream answer accuracy and tool cost.

RAG post-training may update the generator, retriever, query policy, reranker, or several components. Identify which parameters receive supervision. A generator improvement can come from better evidence selection without changing its underlying reasoning.

Tool selection and sequencing are learned decisions. Tool abstention matters when the model already has sufficient information or when no available tool can answer the question. A policy rewarded simply for tool use can over-query.

#### 69.2 Cost-aware objectives and diagnostics

One objective is $\mathbb E[R_{\mathrm{answer}}-\lambda C_{\mathrm{tools}}]$. Another imposes a tool budget and maximizes accuracy within it. Report answer quality, evidence support, latency, and calls separately.

Toolformer is a primary example of learning tool-use behavior from generated and filtered API-call data; its construction provides one case to compare with explicit agent RL. [Toolformer][toolformer]

**Worked example.** A zoning assistant retrieves a relevant parent section but misses an exception in a cross-reference. Potential interventions include query expansion, graph traversal, reranking, or generator training. A single final reward cannot identify which component is responsible without additional diagnostics.

**Exercise.** Use the local search environment to compare a fixed query, learned query rewriting, and an oracle retriever. The oracle gap estimates how much room retrieval leaves before generator quality becomes the main limit.

<a id="chapter-70"></a>
### Chapter 70 — Continual and online learning

#### 70.1 The target distribution moves

Continual SFT updates on new demonstrations. Continual preference learning updates on new judgments. Online RL updates from ongoing interaction. All face nonstationarity: tasks, users, tools, and reward definitions can change.

Forgetting can be measured with a matrix: evaluate each checkpoint on tasks from every previous phase. Replay mixes past examples with current ones, trading retention against adaptation. The replay distribution is itself a design choice.

#### 70.2 Off-policy correction is not a universal repair

Stored trajectories come from older policies. Importance sampling can correct known action-distribution mismatch under support and variance conditions. It does not correct a changed environment, reward definition, or missing observations by itself.

V-trace in IMPALA is an example of clipped importance-weighted value targets designed for distributed actor-learner lag. It illustrates a principled correction with explicit bias/variance choices, not a drop-in proof for every language-agent replay scheme. [IMPALA][impala]

Monitor deployment drift, calibration, and failures by cohort. Keep rollback and evaluation boundaries clear. A continual-learning system needs a decision rule for when adaptation is supported by enough evidence.

**Exercise.** Change the correct tool mapping halfway through the toy environment. Compare full retraining, recent-only updates, and replay mixtures. Measure adaptation speed and retention when the original mapping returns.

<a id="chapter-71"></a>
### Chapter 71 — Meta-learning and automatic curriculum generation

#### 71.1 Learn how to adapt

Meta-learning optimizes performance after adaptation across a task distribution. A simple gradient-based formulation is

$$
\theta'_{\mathcal T}
=\theta-\alpha\nabla_\theta L_{\mathcal T}^{\mathrm{train}}(\theta),
\qquad
\min_\theta\mathbb E_{\mathcal T}
L_{\mathcal T}^{\mathrm{test}}(\theta'_{\mathcal T}).
$$

Differentiating through the inner update introduces second-order terms unless approximated. The split is within each task, and evaluation should include held-out tasks. [Model-Agnostic Meta-Learning][maml]

In-context adaptation and weight-update adaptation are different mechanisms. A policy adapting through memory without changing weights is not the same experiment as meta-training an initialization for gradient descent.

#### 71.2 Curriculum as another learning problem

A teacher can choose tasks based on estimated difficulty, uncertainty, or learning progress. Maximizing current failure rate can select impossible tasks. Maximizing short-term progress can neglect slow but valuable skills.

Self-play and adversarial task generation create challenges from the learner's weaknesses. Automatic environment generation can vary rules, layouts, and horizons. Validate solvability and prevent the generator from encoding shortcuts.

**Worked example.** A task generator notices the learner always fails 100-step problems and sends only those. Reward remains zero. A target-success-rate curriculum mixes shorter tasks where learning signal exists and gradually expands the horizon.

**Exercise.** Compare uniform sampling, failure-focused sampling, and progress-focused sampling in a finite task family. Evaluate all three on a fixed uniform test distribution.

<a id="chapter-72"></a>
### Chapter 72 — Open research problems

#### 72.1 Turn broad challenges into testable questions

| Area | Unresolved issue | Small experiment that studies a mechanism |
|---|---|---|
| Long-horizon credit | Which early decisions caused a late result? | Counterfactual replay from saved states |
| Sparse rewards | Successful behavior is rarely sampled | Vary initial success probability and group size |
| Oversight | Evaluators may miss advanced errors | Train against a weak checker, audit with a stronger one |
| Process rewards | Local validity and eventual success differ | Separate the two labels on generated traces |
| Exploration | Token diversity may not imply strategy diversity | Track distinct solution programs, not wording |
| Reward hacking | Optimization selects checker blind spots | Measure proxy/true reward under stronger selection |
| RL generalization | Gains may remain inside training families | Hold out rules, lengths, and environments |
| Off-policy stability | Corrections have high variance or bias | Control policy lag in a finite MDP |
| Continual agents | Adaptation can erase useful behavior | Alternate task phases with retention evaluation |
| Self-improvement | Feedback can amplify shared errors | Vary independence of generator and checker |
| Multi-agent learning | Coordination can be brittle or nonstationary | Cross-play with unseen partners |
| Production feedback | Observations are selectively collected | Compare random audits with user-reported failures |
| Alignment under capability growth | Stronger policies can exploit fixed oversight | Increase search budget against one frozen grader |
| Compute efficiency | Better updates can cost more to collect | Match total interactions and wall time |

These experiments test mechanisms. Evidence from a finite task can disprove an unconditional claim or demonstrate a possible failure, but does not establish prevalence at scale.

#### 72.2 A research agenda from your own results

Start with an anomaly you can reproduce: a normalized estimator favors one task group, an agent stops using a needed tool, or a supposedly better verifier reduces true accuracy. Write down several explanations before choosing the most interesting one.

**Exercise.** Select one row. State a falsifiable hypothesis, a null result, a counterexample that would change your mind, and the minimum resources required.

<a id="part-xviii"></a>
## Part XVIII — No-GPU Experimental Track

The practical track has been integrated throughout the course. By this point, Labs 1–25 form a connected research-engineering portfolio. This part consolidates the artifacts and checks readiness for independent research.

### A. The reusable experimental stack

| Layer | Built in labs | What it should make easy |
|---|---|---|
| Probability and objective kernel | 6–13 | Compare exact and sampled gradients |
| Tiny LM and adapters | 1–3 | Inspect tokens, masks, and updates |
| Feedback datasets and reward models | 4–5, 17 | Study noise, calibration, and process labels |
| Executable verification | 14–16 | Separate true correctness from proxy reward |
| Agent environment and trajectories | 18–22 | Replay decisions and study credit |
| Coding tasks | 23–24 | Evaluate patches with independent tests |
| Training services | 25 | Measure staleness, queues, and recovery |

### B. Readiness checks

You should be able to reproduce a saved result from its configuration, identify the exact data split, explain the actor loss denominator, and trace any reward to its verifier version. A result that only works in an unrecorded notebook state is not yet a reusable experiment.

Check every numerical result against an exact oracle when one is available. For a larger model where enumeration is impossible, preserve the small oracle tests as implementation checks.

### C. Scaling deliberately

An A experiment answers whether the mechanism exists in a controlled setting. B asks whether it survives realistic language distributions. C buys faster or larger runs. D studies systems behavior. Choose an extension to resolve a concrete uncertainty.

For example, increasing model size is useful if you suspect a result depends on model capacity. It is less useful if the unresolved question is whether a loss mask excludes tool outputs; that can be answered on a tiny model.

**Portfolio exercise:** select one objective experiment, one agent experiment, and one systems experiment. Write a shared reproduction guide and a table of their assumptions. These become the foundation for the five research labs in Part XIX.

<a id="part-xix"></a>
## Part XIX — Doing Post-Training Research

**Chapter navigation:** [73 · Reading papers like a researcher](#chapter-73) · [74 · Reconstructing and comparing algorithms](#chapter-74) · [75 · Bias, variance, and estimator analysis](#chapter-75) · [76 · Forming research questions and hypotheses](#chapter-76) · [77 · Reproduction, ablation, and scientific evidence](#chapter-77) · [78 · From modification to original research](#chapter-78)

**Learning goals:** reconstruct claims, compare estimators, design decisive experiments, and write an honest research report.

<a id="chapter-73"></a>
### Chapter 73 — Reading papers like a researcher

#### 73.1 Separate the question, method, and evidence

Read the abstract to identify the claim, then locate the exact experiment supporting it. Distinguish motivation from measured results. A motivating example may not be representative, and an architectural diagram may omit important training details.

Build a notation dictionary: prompt distribution, policy versions, reward, baseline, ratio, normalization, and loss. Rewrite every equation in the course's notation. If a symbol's distribution is unspecified, record it as an unresolved implementation question.

#### 73.2 Read appendices and code as evidence

Appendices often contain data filtering, sampling settings, reward scales, and failed ablations. Trace implementation to equations. A changed denominator or sampled subset can materially alter the method.

Distinguish algorithmic novelty from more data, stronger base models, better verifiers, longer rollouts, or systems improvements. All can be valuable contributions; they support different claims.

Create a paper dependency map showing inherited algorithms, datasets, models, and evaluation conventions. Read the source of an important inherited assumption rather than relying on successive paraphrases.

**Exercise.** For one GRPO variant, produce a claim table with columns: claim, supporting figure/table, controlled variables, changed variables, uncertainty, and remaining alternative explanations.

<a id="chapter-74"></a>
### Chapter 74 — Reconstructing and comparing algorithms

#### 74.1 Reduce the method to common primitives

A useful canonical description is

$$
\hat g=
\sum_{i,t}w_{it}\,
f(\rho_{it},\hat A_{it})\,
\nabla_\theta\log\pi_\theta(a_{it}\mid h_{it})
+\text{regularizer contributions},
$$

where the exact form of $f$ depends on whether you are describing a gradient or a differentiable surrogate. Do not substitute a clipped objective into this expression without differentiating it.

List the sampling law, reward, baseline, advantage, ratio, clipping, loss reduction, optimizer, and data refresh. Two methods can share an objective but differ in collection policy or optimization schedule.

#### 74.2 Equivalent formulations and hidden differences

With $G$ samples,

$$
R_i-\frac1{G-1}\sum_{j\neq i}R_j
=\frac{G}{G-1}(R_i-\bar R).
$$

Thus leave-one-out centering and group-mean centering differ by a constant for a fixed realized group before additional normalization. Adding division by group standard deviation changes the relationship.

Likewise, maximizing mean reward and minimizing negative mean reward are equivalent sign conventions; averaging log probabilities instead of summing is a substantive change when lengths vary.

**Exercise.** Compare two implementations that use the same method name. Identify three differences that are merely representational and three that change the estimator or data distribution.

<a id="chapter-75"></a>
### Chapter 75 — Bias, variance, and estimator analysis

#### 75.1 Define the target and enumerate a tiny case

Choose a finite policy and reward where the true gradient is exactly computable. Freeze parameters. Generate many independent estimated gradients. Estimate bias as their mean minus the exact gradient and variance as their sample covariance.

This separates an estimator property from the complex feedback of training. A method with lower per-step variance can still learn worse because its bias points away from a useful direction or its data collection is inefficient.

#### 75.2 Control variates and normalization

For score vector $s=\nabla\log\pi(a)$, the scalar constant baseline minimizing $\mathbb E\|(R-b)s\|^2$ is

$$
b^*=\frac{\mathbb E[R\|s\|^2]}{\mathbb E[\|s\|^2]},
$$

when the denominator is nonzero. The mean reward is not always the variance-optimal baseline.

Clipping importance weights bounds extreme contributions but usually introduces bias. Self-normalized importance sampling,

$$
\hat\mu_{\mathrm{SNIS}}=
\frac{\sum_iw_if_i}{\sum_iw_i},
$$

is generally biased at finite sample size but can be consistent under suitable assumptions. Effective sample size

$$
\mathrm{ESS}=\frac{(\sum_iw_i)^2}{\sum_iw_i^2}
$$

is a diagnostic of weight concentration, not a complete guarantee about estimation error.

For normalized group advantages, analyze the numerator, denominator, and their dependence. Group size and reward distribution both matter. Token-level and sequence-level estimators additionally differ in how they treat trajectory likelihood and length.

#### 75.3 When bias can help

An estimator with small bias and much lower variance can have lower mean squared error and yield more stable optimization. That is an empirical tradeoff to measure. Calling an estimator biased does not settle whether it is useful.

**Worked example.** Weights $(1,1,1,9)$ have ESS $12^2/84\approx1.71$, despite four observations. Most influence comes from one sample.

**Exercise.** Use Appendix A to compare raw REINFORCE, group centering, corrected group centering, and standardized group advantages at fixed logits. Predict the mean gradient before running the experiment.

#### 75.4 Why self-normalized importance sampling is biased

Let $X=\sum_iw_if_i$ and $Y=\sum_iw_i$. The self-normalized estimator is $X/Y$. In general,
$\mathbb E[X/Y]\neq\mathbb E[X]/\mathbb E[Y]$.
The numerator and denominator are correlated, so replacing the expectation of a ratio with a ratio of expectations is invalid.

As sample size grows, both sums can concentrate around their population values, yielding consistency under appropriate support and moment conditions. At finite size, rare large weights can dominate.

For a finite two-action example, let the behavior policy choose the valuable action with probability 0.01 while the target chooses it with probability 0.5. The importance weight for that action is 50. A batch of ten often contains none of it, and occasionally contains one highly influential observation. No normalization can create information about an action never sampled.

**Exercise:** enumerate or simulate this example. Compare ordinary importance sampling, self-normalized importance sampling, and clipped weights as batch size grows. Report bias, variance, and mean squared error separately.

<a id="chapter-76"></a>
### Chapter 76 — Forming research questions and hypotheses

#### 76.1 From an observation to a mechanism

An observation is “outputs got longer.” A mechanism hypothesis is “per-response length normalization weakens the negative gradient on long incorrect responses.” A falsifiable prediction specifies how behavior changes if that normalizer is removed while other choices are fixed.

Confounders include reward scale, sampling temperature, prompt difficulty, filtering, and compute budget. A benchmark gain can be real without establishing the proposed mechanism.

#### 76.2 Construct a decisive small experiment

Use a toy problem where only the relevant variables change. Include a boundary case where the mechanism should disappear. For length normalization, fixed-length responses provide such a control.

Derive predictions before seeing results. Define the sign or pattern of the expected effect, not merely “something changes.” A counterexample is especially valuable when it falsifies an overly broad claim.

**Example hypothesis card:**

| Field | Example |
|---|---|
| Observation | A GRPO variant changes the short/long response mixture |
| Hypothesis | Loss reduction reweights actions by response length |
| Intervention | Replace per-response averaging with a fixed denominator |
| Controls | Initial policy, rewards, prompts, samples, learning rate |
| Prediction | Gradient coefficients change by the calculated length factor |
| Boundary condition | No relative change when every response has equal length |
| Disconfirmation | Exact gradients disagree with the predicted weighting |

**Exercise.** Write a second hypothesis explaining the same observation through reward-model length bias. Design an experiment that distinguishes the two.

<a id="chapter-77"></a>
### Chapter 77 — Reproduction, ablation, and scientific evidence

#### 77.1 Decide what is being reproduced

A faithful reproduction matches the original setup closely enough to test its reported result. A scaled-down reproduction of a mechanism changes model size or environment deliberately. Both are useful, but their conclusions differ.

Keep a deviation ledger: model, data, tokenizer, reward, sampling, optimizer, batch construction, hardware, and evaluation. Unknown details are uncertainty, not permission to silently choose whichever variant works best.

#### 77.2 Ablation and sensitivity

An ablation changes one proposed component. Sensitivity analysis varies its value across a range. Hyperparameter fairness means each method has a reasonable opportunity to work under a declared tuning budget.

Negative results should include evidence that the implementation was competent: exact checks, a positive control, and adequate sensitivity. “It failed once” is weak evidence; “it failed across a specified range while the control succeeded” is more informative.

Avoid post-hoc stories. If an explanation was invented after seeing results, label it exploratory and test it in a new experiment. Reproducibility artifacts include data versions, configurations, raw run summaries, analysis code, and environment information.

#### Lab 26 [A; optional B/C] — Paper reproduction

**Objective:** reproduce one post-training result at small scale and document every deviation from the original setup.

Choose a narrow result, such as a baseline identity or a normalization effect. Reconstruct the equations and implementation choices. Run a positive control, reproduce the claimed pattern under your chosen scale, and report both matches and discrepancies.

**Submit:** paper reconstruction, deviation ledger, executable experiment, raw results, and scope statement. **Check:** call a reduced-scale mechanism study exactly that if it cannot reproduce the paper's model/data scale.

#### Lab 27 [A; optional B/C] — Ablation study

**Objective:** hold environment, data, and compute fixed while removing or changing one claimed key component.

Predefine the component, baseline, metric, and resource accounting. Use at least one boundary condition and several seeds. If the component changes sampling efficiency, show both fixed-interaction and fixed-wall-time comparisons.

**Submit:** ablation matrix, uncertainty, implementation checks, and alternative explanations. **Check:** no simultaneous change to reward scale, model initialization, or data filtering unless explicitly part of the experiment.

#### Lab 28 [A; optional B/C] — Estimator study

**Objective:** compare two advantage, baseline, or importance-weighting choices and analyze bias, variance, stability, and sample efficiency.

First compare gradients at fixed parameters against an exact oracle. Then train with matched interactions. Vary group size, reward sparsity, and policy lag one at a time.

**Submit:** bias/variance plots, ESS where relevant, learning curves, and an explanation connecting estimator measurements to training outcomes. **Check:** a lower-variance result alone is insufficient to claim a better optimizer.

<a id="chapter-78"></a>
### Chapter 78 — From modification to original research

#### 78.1 A contribution needs a precise claim

Original research can propose an objective, estimator, reward, curriculum, environment, diagnostic, or counterexample. Novelty requires checking related work. A renamed existing formula is not a new contribution.

A theoretical contribution may prove an identity, bound, or impossibility under assumptions. An empirical contribution can establish a robust mechanism, failure mode, or useful method through controlled evidence. Strong work states what the evidence does not establish.

#### 78.2 Build a minimal research report

Start with the question and why it matters. Describe prior work, method, assumptions, experiments, results, limitations, negative findings, and next tests. Every plot should support a specific claim.

Scaling claims require scaling evidence. A finite-policy counterexample can refute an unconditional assertion, while a toy performance improvement usually cannot justify a frontier-performance claim.

#### Lab 29 [A; optional B/C] — Original research mini-project

**Objective:** form a falsifiable hypothesis about a post-training or agent-training mechanism, design the smallest convincing experiment, and test it.

Select a question from Chapter 72 or an anomaly in your own labs. Conduct a focused literature search. Write a hypothesis card, design an exact or controlled setting, implement a baseline, and predefine the evaluation. Run a pilot, revise implementation errors, then execute the declared experiment.

**Submit:** research question, related-work map, derivation or prediction, experiment, results, uncertainty, and a proposed follow-up. **Success criterion:** credible evidence for or against the hypothesis, even if the proposed improvement fails.

#### Lab 30 [A] — Research write-up

**Objective:** produce a short paper-style report with motivation, related work, method, experimental design, results, limitations, negative findings, and next experiments.

Write a report that another researcher can audit without your explanation. Include a reproducibility appendix with exact configurations and artifact locations. Ask whether each claim is supported by a figure, table, derivation, or explicit observation.

**Submit:** the report and reproduction package. **Final review:** remove unsupported novelty claims, distinguish exploratory from confirmatory analysis, and include the strongest evidence against your favored interpretation.

### End-of-course standards

**Research Engineer standard:** for a new method, identify its sampling distribution, reward, baseline, advantage, ratio, KL treatment, gradient estimator, failure modes, implementation details, rollout infrastructure, and fair evaluation. For an agent, specify the harness-induced MDP/POMDP, decision boundaries, trajectory data, credit assignment, verifier, and loss. Implement, debug, reproduce, and explain scaling constraints.

**Researcher standard:** reconstruct the method from first principles, identify assumptions, analyze bias and variance, distinguish algorithmic effects from data/compute/system changes, form alternative hypotheses, design decisive ablations, and produce small but rigorous original investigations.

These standards are demonstrated through work. Completing the reading is the beginning of that evidence; the experiments and reports make it visible.

<a id="appendices"></a>
## Appendices

### Appendix A — A runnable CPU reference kernel

This kernel uses Python and NumPy. It deliberately keeps the response space finite, so exact sums are available. That makes it useful for Labs 6–13 and 28. It is a mathematical reference, not a scalable LM trainer.

Copy the following block into a Python file and run it. The checks compare mathematical identities and finite-difference gradients. Training assignments extend this code; they should preserve its exact tests.

```python
import math
import json
import numpy as np


def log_softmax(z):
    z = np.asarray(z, dtype=np.float64)
    shifted = z - np.max(z, axis=-1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))


def sigmoid(z):
    # Scalar form with stable behavior for either sign.
    z = float(z)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def objective_and_gradient(logits, rewards, reference, beta=0.0):
    """Exact J = E[R] - beta * KL(policy || reference)."""
    logits = np.asarray(logits, dtype=np.float64)
    rewards = np.asarray(rewards, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if logits.ndim != 1 or logits.shape != rewards.shape:
        raise ValueError("Expected one reward for each categorical action.")
    if reference.shape != logits.shape or np.any(reference <= 0):
        raise ValueError("Reference must have positive matching support.")
    if not np.isclose(reference.sum(), 1.0):
        raise ValueError("Reference probabilities must sum to one.")
    logp = log_softmax(logits)
    p = np.exp(logp)
    augmented = rewards - beta * (logp - np.log(reference))
    objective = float(p @ augmented)
    gradient = p * (augmented - objective)
    return objective, gradient


def finite_difference(function, z, step=1e-6):
    z = np.asarray(z, dtype=np.float64)
    result = np.zeros_like(z)
    for j in range(z.size):
        direction = np.zeros_like(z)
        direction[j] = step
        result[j] = (function(z + direction) -
                     function(z - direction)) / (2.0 * step)
    return result


def sample_gradients(logits, rewards, groups, group_size, method, seed=0):
    """One gradient estimate per independently sampled response group.

    Methods: reinforce, centered, rloo, standardized.
    No clipping, KL, or token normalization is included here.
    """
    if groups < 1 or group_size < 1:
        raise ValueError("Positive sample counts required.")
    p = np.exp(log_softmax(logits))
    rewards = np.asarray(rewards, dtype=np.float64)
    rng = np.random.default_rng(seed)
    actions = rng.choice(len(p), size=(groups, group_size), p=p)
    r = rewards[actions]
    score = np.eye(len(p))[actions] - p
    if method == "reinforce":
        advantages = r
    elif method == "centered":
        advantages = r - r.mean(axis=1, keepdims=True)
    elif method == "rloo":
        if group_size < 2:
            raise ValueError("RLOO requires at least two samples.")
        baseline = (r.sum(axis=1, keepdims=True) - r) / (group_size - 1)
        advantages = r - baseline
    elif method == "standardized":
        centered = r - r.mean(axis=1, keepdims=True)
        scale = r.std(axis=1, keepdims=True, ddof=0)
        advantages = centered / (scale + 1e-8)
    else:
        raise ValueError("Unknown estimator.")
    return (advantages[..., None] * score).mean(axis=1)


def dpo_loss_and_gradient(logits, reference, winner, loser, beta):
    logp = log_softmax(logits)
    logq = np.log(np.asarray(reference, dtype=np.float64))
    margin = beta * ((logp[winner] - logq[winner]) -
                     (logp[loser] - logq[loser]))
    loss = float(np.logaddexp(0.0, -margin))
    gradient = np.zeros_like(logp)
    coefficient = beta * (sigmoid(margin) - 1.0)
    gradient[winner] += coefficient
    gradient[loser] -= coefficient
    return loss, gradient


def ppo_surrogate(ratios, advantages, epsilon=0.2):
    ratios = np.asarray(ratios, dtype=np.float64)
    advantages = np.asarray(advantages, dtype=np.float64)
    clipped = np.clip(ratios, 1.0 - epsilon, 1.0 + epsilon)
    return np.minimum(ratios * advantages, clipped * advantages)


def gae(rewards, values, dones, gamma=1.0, lam=0.95):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=bool)
    if len(values) != len(rewards) + 1 or len(dones) != len(rewards):
        raise ValueError("Need T rewards/dones and T+1 values.")
    advantages = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        alive = 0.0 if dones[t] else 1.0
        delta = rewards[t] + gamma * alive * values[t + 1] - values[t]
        running = delta + gamma * lam * alive * running
        advantages[t] = running
    return advantages, advantages + values[:-1]


def grade_integer_response(text, expected):
    """A small strict checker for the synthetic arithmetic environment."""
    if not isinstance(text, str):
        return {"status": "type_error", "reward": 0.0}
    if len(text) > 200:
        return {"status": "too_long", "reward": 0.0}
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON key.")
            result[key] = value
        return result
    try:
        data = json.loads(text, object_pairs_hook=unique_object)
    except (TypeError, ValueError):
        return {"status": "parse_error", "reward": 0.0}
    if not isinstance(data, dict) or set(data) != {"answer"}:
        return {"status": "schema_error", "reward": 0.0}
    if type(data["answer"]) is not int:  # Reject bool as well as strings.
        return {"status": "type_error", "reward": 0.0}
    return {
        "status": "correct" if data["answer"] == expected else "incorrect",
        "reward": float(data["answer"] == expected),
    }


def reference_checks():
    z = np.array([-0.7, 0.2, 0.4])
    rewards = np.array([0.0, 0.5, 1.0])
    reference = np.array([0.2, 0.5, 0.3])
    beta = 0.3
    _, analytic = objective_and_gradient(z, rewards, reference, beta)
    numerical = finite_difference(
        lambda v: objective_and_gradient(v, rewards, reference, beta)[0], z)
    assert np.allclose(analytic, numerical, atol=1e-7)
    _, analytic_dpo = dpo_loss_and_gradient(z, reference, 2, 0, beta)
    numerical_dpo = finite_difference(
        lambda v: dpo_loss_and_gradient(v, reference, 2, 0, beta)[0], z)
    assert np.allclose(analytic_dpo, numerical_dpo, atol=1e-7)

    # The KL-regularized distribution-space optimum has zero logit gradient.
    optimum_logits = np.log(reference) + rewards / beta
    _, optimum_gradient = objective_and_gradient(
        optimum_logits, rewards, reference, beta)
    assert np.allclose(optimum_gradient, 0.0, atol=1e-12)

    assert np.allclose(ppo_surrogate([1.4, 0.6], [2.0, -2.0]), [2.4, -1.6])
    adv, targets = gae([-0.1, 0.8], [0.4, 0.5, 0.0],
                       [False, True], gamma=1.0, lam=1.0)
    assert np.allclose(adv, [0.3, 0.3])
    assert np.allclose(targets, [0.7, 0.8])

    group_rewards = np.array([0.0, 1.0, 1.0])
    loo = group_rewards - (group_rewards.sum() - group_rewards) / 2
    assert np.allclose(loo, [-1.0, 0.5, 0.5])
    assert grade_integer_response('{"answer": 5}', 5)["reward"] == 1.0
    assert grade_integer_response('{"answer": true}', 1)["reward"] == 0.0
    assert grade_integer_response('{"answer": 4, "answer": 5}', 5)["reward"] == 0.0
    assert grade_integer_response('wrong 5 right', 5)["reward"] == 0.0
    print("Exact objective, DPO, PPO, GAE, RLOO, and verifier checks passed.")


if __name__ == "__main__":
    reference_checks()
```

#### A.1 How to use the kernel for an estimator experiment

Choose logits and reward values. Compute the exact unregularized gradient with beta zero. Call the sampled estimator for many independent groups without changing logits. Compare the mean estimate with the exact gradient.

For group centering with fixed group size $G$, compare its mean with $(G-1)/G$ times the exact gradient. For RLOO, compare with the exact gradient itself. For standardized advantages, do not assume that either equality holds.

Use the empirical covariance across independent groups to estimate variance. When comparing trained policies later, keep this fixed-parameter result separate from learning curves.

#### A.2 A minimal training loop

```python
# Run after the previous block, or import its functions.
logits = np.zeros(3, dtype=np.float64)
rewards = np.array([0.0, 0.5, 1.0])
reference = np.ones(3) / 3.0
beta = 0.4
for update in range(400):
    objective, gradient = objective_and_gradient(
        logits, rewards, reference, beta)
    logits += 0.2 * gradient
    logits -= logits.mean()  # Remove the unidentifiable common offset.
learned = np.exp(log_softmax(logits))
expected = np.exp(log_softmax(np.log(reference) + rewards / beta))
print("learned:", learned)
print("analytic optimum:", expected)
```

This loop uses the exact gradient, so it is a control condition. Replace it with a sampled estimator only after checking the fixed-parameter mean. Do not compare algorithms using different hidden regularizers or reward definitions.

#### A.3 Tiny autoregression without a deep-learning framework

The following bigram LM is a CPU starting point for Lab 1. It predicts a token from the immediately previous token, so it cannot solve tasks requiring arbitrary long context. That limitation is deliberate: verify the data path, then replace the table with a recurrent network or causal transformer.

```python
class BigramLM:
    def __init__(self, vocabulary_size, seed=0):
        rng = np.random.default_rng(seed)
        self.logits = rng.normal(0.0, 0.01,
                                 (vocabulary_size, vocabulary_size))

    def loss_and_gradient(self, inputs, targets, mask):
        inputs = np.asarray(inputs, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.int64)
        mask = np.asarray(mask, dtype=np.float64)
        if inputs.shape != targets.shape or mask.shape != targets.shape:
            raise ValueError("Inputs, targets, and mask must align.")
        denominator = mask.sum()
        if denominator <= 0:
            raise ValueError("At least one target token is required.")
        logp = log_softmax(self.logits[inputs])
        p = np.exp(logp)
        token_logp = np.take_along_axis(
            logp, targets[..., None], axis=-1)[..., 0]
        loss = -float((token_logp * mask).sum() / denominator)
        derivative = p.copy()
        flat_derivative = derivative.reshape(-1, derivative.shape[-1])
        flat_targets = targets.reshape(-1)
        flat_derivative[np.arange(flat_targets.size), flat_targets] -= 1.0
        derivative *= (mask / denominator)[..., None]
        gradient = np.zeros_like(self.logits)
        np.add.at(gradient, inputs.reshape(-1),
                  derivative.reshape(-1, derivative.shape[-1]))
        return loss, gradient

    def sample(self, start_id, eos_id, max_new_tokens=20, seed=0):
        rng = np.random.default_rng(seed)
        output = [start_id]
        for _ in range(max_new_tokens):
            p = np.exp(log_softmax(self.logits[output[-1]]))
            token = int(rng.choice(len(p), p=p))
            output.append(token)
            if token == eos_id:
                break
        return output


def bigram_checks():
    model = BigramLM(4)
    x = np.array([[0, 1, 2], [0, 2, 1]])
    y = np.array([[1, 2, 3], [2, 1, 3]])
    mask = np.array([[0, 1, 1], [0, 1, 1]], dtype=float)
    loss, grad = model.loss_and_gradient(x, y, mask)
    before = model.logits.copy()
    flat = before.reshape(-1)

    def flattened_loss(v):
        model.logits = v.reshape(before.shape).copy()
        return model.loss_and_gradient(x, y, mask)[0]

    numeric = finite_difference(flattened_loss, flat)
    model.logits = before
    assert np.allclose(grad.reshape(-1), numeric, atol=1e-7)

    # Adding a masked padding column must preserve loss and gradient.
    padded_x = np.pad(x, ((0, 0), (0, 1)))
    padded_y = np.pad(y, ((0, 0), (0, 1)))
    padded_mask = np.pad(mask, ((0, 0), (0, 1)))
    padded_loss, padded_grad = model.loss_and_gradient(
        padded_x, padded_y, padded_mask)
    assert np.isclose(loss, padded_loss)
    assert np.allclose(grad, padded_grad)
    print("Bigram gradient and padding checks passed.")


bigram_checks()
```

**Extension path:** replace the lookup table with an embedding, a recurrent or attention block, and a vocabulary projection. Keep the exact same input/target shift and mask tests. Add a causality test that changes future tokens while comparing earlier logits.

### Appendix B — Worked solutions and self-assessment

These solutions explain representative exercises. Open research exercises have no predetermined winning result.

#### B.1 Probability, KL, and the policy gradient

**Sequence score.** For probabilities $0.5,0.2,0.8$, multiplication gives 0.08. The logarithm converts multiplication into addition. A token average divides the score by three and therefore changes comparisons across lengths.

**Baseline cancellation.** Conditional on a state, a baseline independent of the action factors out of the sum. The remaining sum is the derivative of total probability, which is zero. If the baseline uses that action's own reward, this argument no longer applies without correction.

**GAE.** With residuals $0.2,-0.1,0.6$, $\gamma=1,\lambda=0.5$: the last advantage is 0.6; the previous one is $-0.1+0.5(0.6)=0.2$; the first is $0.2+0.5(0.2)=0.3$.

**Trust regions.** PPO's flat surrogate region removes the gradient incentive from a particular sample in one direction. Shared parameters can still move the action probability because of other samples or losses. This is why clipping is not a hard KL constraint.

#### B.2 SFT, LoRA, and imitation

**Masking.** A system or user token can be necessary context without receiving a target loss. Excluding it from attention would remove information; excluding it only from the loss avoids training the model to reproduce it.

**LoRA initialization.** For $BA$, the gradient with respect to $B$ contains $A$, and the gradient with respect to $A$ contains $B$. If both are zero, both gradients are zero. Initializing one randomly and one to zero permits one factor to move first.

**Compounding errors.** The simple independent-decision model gives $0.98^{50}\approx0.364$. It is an illustration, not a universal bound. A real agent may recover from mistakes or make correlated errors, changing the outcome.

**Distillation coverage.** Teacher demonstrations only at teacher states omit learner mistakes. Teacher corrections on student states directly provide information in the distribution where the student needs help.

#### B.3 Reward learning and preference optimization

**Reward identifiability.** Pairwise probability depends on $r_w-r_l$. Adding a constant to every candidate for one prompt leaves all comparisons unchanged. An absolute score threshold cannot be interpreted without further calibration.

**DPO.** Substituting $R=\beta\log(\pi^*/\pi_{\mathrm{ref}})+\beta\log Z$ into a reward difference cancels the partition term. This removes the need to compute $Z$ in the pair loss; it does not eliminate finite-data uncertainty.

**Unobserved responses.** A dataset comparing only A against B does not directly tell the learner whether C is excellent or terrible. The model's parameterization, initialization, and regularization determine behavior there.

#### B.4 Modern RL and verification

**RLOO versus group centering.** Algebra gives $A_i^{\mathrm{LOO}}=G(R_i-\bar R)/(G-1)$. This constant relationship holds before dividing by a sample standard deviation or applying different clipping/reduction rules.

**Group informativeness.** For binary reward, a group is flat if every sample fails or every sample succeeds. These events have probabilities $(1-p)^G$ and $p^G$. Subtracting them from 1 gives the mixed-group probability.

**Verifier robustness.** A substring checker can pass a response that includes every possible answer. A strict schema and task-specific comparison reduce this loophole. Independent hidden cases test whether the checker has other gaps.

**Reasoning claims.** A longer trace is evidence of more generated computation, not automatically of better reasoning. Causal intervention and matched-budget evaluation provide stronger evidence.

#### B.5 Agents, shaping, and evaluation

**Causal trace construction.** At decision $k$, include only observations available before that action. A future tool result is leakage even if it appears naturally in the complete stored episode.

**Potential shaping.** Adjacent terms cancel in the discounted sum. Remaining terms are a start-state constant and a terminal term. If the terminal term depends on which outcome the policy reaches, policy ordering may change.

**Pass@k.** For five samples with two successes, there are ten subsets of size two and three all-failure subsets. Thus the estimate is $1-3/10=0.7$.

**Paired comparisons.** Resampling task pairs preserves the correlation between model outcomes on the same task. Resampling each model independently loses that information and can distort uncertainty.

#### B.6 Systems and research

**KV memory.** Count key and value tensors for each layer, cached token, active sequence, and KV head. The 0.75-GiB example excludes weights, activations, and runtime overhead.

**Policy staleness.** A sample generated by version 8 does not become on-policy merely because version 10 is now deployed. Its behavior probabilities remain those of version 8.

**Estimator bias.** Bias is relative to a specified target. A clipped estimator may intentionally exchange bias for lower variance. Compare its expected direction and mean squared error before interpreting training curves.

**Original contribution.** A reproducible counterexample can be a useful finding. A proposed improvement that fails is still informative if implementation checks, controls, and uncertainty support the negative result.

### Appendix C — Figure register

All visual placeholders are optional illustrations; no lesson depends on an unavailable image.

| ID | Placement | Required content | Suggested format |
|---|---|---|---|
| F01 | Chapter 0.3 | Collection, training, and evaluation loops with version boundaries | Vector flow diagram |
| F02 | Chapter 5 | PPO surrogate versus ratio for positive/negative advantage | Two-panel mathematical plot |
| F03 | Chapter 6 | Token IDs, shifted targets, context mask, target mask | Annotated table |
| F04 | Chapter 25 | Mixed, all-failure, and all-success reward groups | Grouped dot plot with numbers |
| F05 | Chapter 34 | Search tree and selected supervision | Small tree diagram |
| F06 | Chapter 42 | Early decision, late outcome, counterfactual branch | Trajectory diagram |
| F07 | Chapter 58 | Seed curves and uncertainty at matched rollout budget | Scientific line plot |
| F08 | Chapter 60 | Workers, trajectory store, weights, queues, staleness | Systems diagram |

For F02, plot directly from the equation. For F07, use measured lab results rather than illustrative fabricated curves. All figures should state units, legends, and whether values are measured or schematic.

### Appendix D — Primary-source reading shelf

Read sources after working through the relevant lesson. Start by reconstructing one equation and one experiment. Record the paper version and implementation commit when reproducing a method.

| Theme | Sources | Reading task |
|---|---|---|
| Trust regions and advantages | [TRPO][trpo], [PPO][ppo], [GAE][gae] | Map assumptions to practical approximations |
| Efficient adaptation | [LoRA][lora], [QLoRA][qlora] | Identify trainable state and numerical representations |
| Imitation | [DAgger][dagger] | Explain the state-distribution mismatch |
| Human feedback | [InstructGPT][instructgpt], [Process supervision][prm] | Separate label type from optimization method |
| Preference objectives | [DPO][dpo], [IPO][ipo], [KTO][kto], [SimPO][simpo] | Rewrite each objective under common notation |
| Further preference methods | [ORPO][orpo], [CPO][cpo], [BCO][bco], [NCA][nca], [SLiC-HF][slic], [RRHF][rrhf], [Nash learning][nash] | Compare score, loss, reference, and feedback format |
| Critic-free RL | [RLOO][rloo], [ReMax][remax], [DeepSeekMath/GRPO][grpo] | Analyze baseline dependence |
| Modern variants | [Dr. GRPO][drgrpo], [REINFORCE++][reinforcepp], [DAPO][dapo], [GSPO][gspo] | Isolate ratio, normalization, and sampling changes |
| Reasoning pipeline | [DeepSeek-R1][r1] | Trace data lineage across stages |
| Agents | [ReAct][react], [WebArena][webarena], [SWE-bench][swebench], [Agent Lightning][lightning] | Reconstruct the environment and training boundary |
| Rewards and evaluation | [Potential shaping][shaping], [HumanEval/pass@k][humaneval], [LLM judging][judge] | State what each measurement supports |
| Systems | [ZeRO][zero], [PagedAttention][pagedattention], [HybridFlow][hybridflow], [IMPALA][impala] | Connect architecture to estimator assumptions |
| Alignment | [Constitutional AI][constitutional] | Audit feedback independence and behavioral criteria |
| Specialized topics | [LLaVA][llava], [Ring Attention][ring], [Toolformer][toolformer], [MAML][maml] | Transfer the course framework to a new setting |

[trpo]: https://arxiv.org/abs/1502.05477
[ppo]: https://arxiv.org/abs/1707.06347
[gae]: https://arxiv.org/abs/1506.02438
[lora]: https://arxiv.org/abs/2106.09685
[qlora]: https://arxiv.org/abs/2305.14314
[dagger]: https://proceedings.mlr.press/v15/ross11a.html
[instructgpt]: https://arxiv.org/abs/2203.02155
[prm]: https://arxiv.org/abs/2305.20050
[dpo]: https://arxiv.org/abs/2305.18290
[ipo]: https://arxiv.org/abs/2310.12036
[kto]: https://arxiv.org/abs/2402.01306
[simpo]: https://arxiv.org/abs/2405.14734
[orpo]: https://arxiv.org/abs/2403.07691
[cpo]: https://arxiv.org/abs/2401.08417
[bco]: https://arxiv.org/abs/2404.04656
[nca]: https://arxiv.org/abs/2402.05369
[slic]: https://arxiv.org/abs/2305.10425
[rrhf]: https://arxiv.org/abs/2304.05302
[nash]: https://arxiv.org/abs/2312.00886
[rloo]: https://arxiv.org/abs/2402.14740
[remax]: https://arxiv.org/abs/2310.10505
[grpo]: https://arxiv.org/abs/2402.03300
[drgrpo]: https://arxiv.org/html/2503.20783v2
[reinforcepp]: https://arxiv.org/abs/2501.03262
[dapo]: https://arxiv.org/html/2503.14476v2
[gspo]: https://arxiv.org/html/2507.18071v2
[r1]: https://arxiv.org/abs/2501.12948
[react]: https://arxiv.org/abs/2210.03629
[webarena]: https://arxiv.org/abs/2307.13854
[swebench]: https://arxiv.org/abs/2310.06770
[lightning]: https://arxiv.org/abs/2508.03680
[shaping]: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
[humaneval]: https://arxiv.org/abs/2107.03374
[judge]: https://arxiv.org/abs/2306.05685
[zero]: https://arxiv.org/abs/1910.02054
[pagedattention]: https://arxiv.org/abs/2309.06180
[hybridflow]: https://arxiv.org/abs/2409.19256
[impala]: https://arxiv.org/abs/1802.01561
[constitutional]: https://arxiv.org/abs/2212.08073
[llava]: https://arxiv.org/abs/2304.08485
[ring]: https://arxiv.org/abs/2310.01889
[toolformer]: https://arxiv.org/abs/2302.04761
[maml]: https://arxiv.org/abs/1703.03400

---

**The next action:** begin Chapter 0.1, write the Part 0 system specification, and then work through the finite-policy derivations in Part I. Return to this manuscript as both a sequence of lessons and a reference while completing the labs.
