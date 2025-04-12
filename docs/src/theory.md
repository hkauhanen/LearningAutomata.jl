# Theory

## General definitions

Suppose a learner has access to $n$ actions ``\mathcal{A}_1, \dots , \mathcal{A}_n`` and chooses the $i$th action with probability $W_i$, the corresponding *action weight*, a random variable. We collect the individual action probabilities into a vector $\mathbf{W} = (W_1, \dots , W_n) \in \Delta$ and call this the learner's current *knowledge state*. The set

```math
\Delta = \Delta^{n-1} = \{\mathbf{x} \in \mathbb{R}^n : x_i \geq 0 \textnormal{ and } \textstyle\sum_i x_i = 1\}
```

is the $(n-1)$-dimensional simplex, which we can regard as the learner's *state space*.

Having chosen an action, the learner interacts with a *learning environment* which either *rewards* or *punishes* the action. Mathematically, for each action $\mathcal{A}_i$, there exists both a *reward operator* $u_i^+: \Delta \to \Delta$ with the property that $u_i^+(\mathbf{W})_i > W_i$ (i.e. the weight $W_i$ increases), as well as a *punishment operator* $u_i^-: \Delta \to \Delta$ with the property that $u_i^-(\mathbf{W})_i < W_i$.

The cycle continues: the learner again chooses an action (now using the updated $\mathbf{W}$) and the environment responds with either reward or punishment.


## Learning environments

In most cases, it is assumed that the learning environment is a *stationary random environment* (SRE). This means that the environment can be characterized by a set of constant *penalty probabilities*: we write $c_i$ for the probability that action $\mathcal{A}_i$ is punished by the environment. (Moreover, since we assume no third possible environmental response, it follows that $1-c_i$ is the probability of a reward.)

A stationary random environment will be called *omnipunitive* if $c_i > 0$ for all $i$, in other words, if the environment punishes each possible action at least some of the time.

Much is known about the expected behaviour of learners in stationary random environments; some of these results will be summarized in what follows. The true usefulness of LearningAutomata.jl, however, lies in the fact that learning environments of arbitrary complexity -- in particular, environments made up of a set of other learning agents -- can be simulated. This is expounded on in more detail in the section on Agents.jl integrationref.


## Linear reward--penalty learning with two actions

The choice of the learning operators $u_i^{\pm}$ constitutes the *learning algorithm*. A classical choice is the *linear reward--penalty* (LRP) scheme [BushMosteller1955](@cite). With two actions ($n=2$), this amounts to adopting the following four operators:

FIXME

Here, the $0 < \gamma_i, \beta_i < 1$ are *learning rate* parameters which control the magnitude of reward ($\gamma_i$) or penalty ($\beta_i$) made to the action weights. It is usually assumed that they coincide, so that we can write $\theta = \gamma_1 = \beta_1 = \gamma_2 = \beta_2$ for some $0 < \theta < 1$.

Of particular interest is how the expected weight vector $\mathbb{E}\mathbf{W}$ evolves as learning continues. With LRP and two actions in a stationary random environment characterized by penalty probabilities $c_1 > 0$ and $c_2 > 0$, it is known that, with increasing learning iteration $t$, $\mathbb{E}W_1(t)$ tends to the limit

$$\lim_{t\to \infty} \mathbb{E} W_1(t) = \frac{c_2}{c_1 + c_2}.$$

Correspondingly, $\mathbb{E}W_2(t)$ tends to

$$\lim_{t \to \infty} \mathbb{E} W_2(t) = \frac{c_1}{c_1 + c_2}.$$

(This is necessary, since we must have $W_1 + W_2 = 1$.)

Note that the limits do not depend on the magnitude(s) of the learning rate parameter(s).


## Linear reward--penalty learning with $n$ actions

Generalizing to $n$ actions, the LRP scheme takes the following form. Suppose $\mathcal{A}_k$ is the action chosen by the learner. Then the operators are:

FIXME

If such a learner is exposed to an omnipunitive SRE ($c_i > 0$ for all $i$), then it can be shown that the expected value $\mathbb{E}W_i$ tends to

$$\lim_{t \to \infty} \mathbb{E}W_i(t) = \frac{c_i^{-1}}{\sum_{j=1}^n c_j^{-1}}$$

with increasing learning iteration [NarendraThathachar1989](@cite). In other words, the expected weight on action $\mathcal{A}_i$ is proportional to the inverse of the penalty on $\mathcal{A}_i$, normalized across all actions.


## Mean learning dynamic

It is often interesting to consider the conditional expectation $\mathbb{E}[\mathbf{W}(t) \mid \mathbf{W}(0)]$, i.e. the expected weight vector at learning iteration $t$ given the initial starting point. For economy of notation, let us fix the initial state at $\mathbf{w}_0 \in \Delta$ and write $\mathbf{v}(t) = \mathbb{E}[\mathbf{W}(t) \mid \mathbf{W}(0) = \mathbf{w}_0]$.

To study how $\mathbf{v}(t)$ evolves, we may note that, in general (i.e. for any learning scheme),

$$v_i(t+1) = \sum_{j=1}^n W_j(t) \bigg( c_j u_j^- \big(\mathbf{W}(t)\big)_i + (1-c_j) u_j^+\big(\mathbf{W}(t)\big)_i \bigg)$$

assuming the learning environment is stationary. Taking expectations on both sides, we further obtain

$$v_i(t+1) = \sum_{j=1}^n v_j(t) \bigg( c_j u_j^- \big(\mathbf{v}(t)\big)_i + (1-c_j) u_j^+\big(\mathbf{v}(t)\big)_i \bigg)$$

For particular choices of the learning operators $u_j^{\pm}$, this *mean (learning) dynamic* may turn out to have a particularly simple form. For instance, with LRP with a common learning rate parameter $\theta$, it can be shown that

$$\mathbf{v}(t+1) = B\mathbf{v}(t)$$

where the matrix $B = [b_{ij}]$ has

$$b_{ij} = \begin{cases} 1 - \theta c_i & \textnormal{if } i = j \\ \frac{\theta}{n-1} c_j & \textnormal{if } i \neq j\end{cases}$$

in cell $(i,j)$.


## Fluid limit

It is often possible to approximate learning trajectories using an ordinary differential equation (ODE). Suppose that learning opportunities arrive at times $\tau_1, \tau_2, \tau_3, \ldots \in \mathbb{R}$, where each such time is an exponentially distributed random variable with rate $\lambda$. If $\lambda \to \infty$ while $\theta \to 0$ such that the product $\theta \lambda$ remains bounded, then we have (assuming LRP in a SRE)

$$\dot{\mathbf{v}} = B^* \mathbf{v}$$

where $B^* = [b^*_{ij}]$ has

$$b^*_{ij} = \begin{cases} - \theta^* c_i & \textnormal{if } i = j \\ \frac{\theta^*}{n-1} c_j & \textnormal{if } i \neq j\end{cases}$$

in cell $(i,j)$. Here, $\theta^* = \theta / \mathbb{E}\tau$.

Individual learning trajectories (individual realizations of the stochastic process $\{\mathbf{W}(t)\}_{t \geq 0}$) dance around the solution of this ODE, sticking closer and closer to it as the learning rate is diminished. This has been called the *slow-learning limit* in some previous literature (REF), though we will mostly stick to the more general term *fluid limit*.

LearningAutomata.jl provides methods for computing the mean learning dynamic and the associated fluid limit for the LRP algorithm; see REF for an illustration.


## Games

Suppose two learners meet, one employing action $\mathcal{A}_i$ and the other employing action $\mathcal{A}_j$. How should each learner modify their knowledge state after this encounter?

Let us write $a_{ij}$ for the probability that such an interaction results in a penalty for the first learner. In other words, the quantity $a_{ij}$ can be thought of as the probability with which action $\mathcal{A}_j$ punishes action $\mathcal{A}_i$.

We collect these quantities in a matrix,

$$A = [a_{ij}] = \begin{pmatrix} a_{11} & a_{12} & \dots & a_{1n} \\ a_{21} & a_{22} & \dots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{n1} & a_{n2} & \dots & a_{nn} \end{pmatrix},$$

known as an *advantage matrix*. (In a sense, this is the inverse of a payoff matrix.)

Assuming a population of sufficiently well mixing learners, the penalty probability for action $\mathcal{A}_i$ may now be expressed as follows:

$$c_i = c_i(\mathbf{x}) = \sum_{j=1}^n a_{ij} x_j = (A \mathbf{x})_i,$$

where $x_j$ is the probability of encountering a learner that employs action $\mathcal{A}_j$.

Evidently, the learning environment is no longer stationary as the penalties change as the population composition $\mathbf{x} = (x_1, \ldots , x_n)$ changes. However, in some cases it is still possible to write a mean dynamic and fluid limit for $\mathbf{x}$, study how this evolves, and compare the realizations of the full stochastic process against those deterministic predictions. LearningAutomata.jl provides some tools to facilitate this in the case in which the individual learners employ LRP.


## Action costs

In certain applications, it makes sense to assume that the mere act of employing a particular action leads to some amount of decrease in that action's weight. In other words, the action may be costly to perform.

It is very straightforward to include such a notion of action cost in the general LRP scheme. Assume $0 \leq \delta_i < 1$ is the cost associated with action $\mathcal{A}_i$.


## Historical remarks and further reading


