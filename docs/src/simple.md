# Learning in simple environments

```@setup simple
using Random
Random.seed!(123)
```

Here, we illustrate the use of LearningAutomata.jl in simple instances in which the learning environment is stationary and the learner employs the linear reward--penalty (LRP) scheme.

!!! note
    See the [Theory](@ref) section for details on the concepts employed here, such
    as "LRP", "learning rate", "advantage matrix", and so on.

First, of course, we need to load the package:

```@example simple
using LearningAutomata
```


## The `LRPLearner` type and its constructors

The central structure provided by LearningAutomata.jl is the `LRPLearner` composite type, representing a learner that updates its knowledge state using the LRP scheme [BushMosteller1955](@cite). Two constructors are provided for this type. The first one takes two positional arguments, one `Int` and one `Float64`:

```@example simple
alice = LRPLearner(2, 0.001)
```

The first argument sets the number of actions, the second argument sets a global learning rate that applies to all actions and both rewards and penalties. The learner's initial state is automatically set to the maximum-entropy state $(1/n, \ldots , 1/n)$, where $n$ is the number of actions.

The second constructor takes, like the first one, the number of actions as the first positional argument. The second positional argument, however, is of type `Vector{Float64}`. This makes it possible to give each action its own learning rate, e.g.:

```@example simple
bob = LRPLearner(3, [0.001, 0.005, 0.02])
```

Each constructor furthermore accepts a number of keyword arguments to further specify the learner's behaviour. These are:

- `β::Vector{Float64}`: a learning rate vector that applies to punishments (the $i$th component of this vector applying to the $i$th action)
- `δ::Vector{Float64}`: a vector of action costs
- `W::Vector{Float64}`: the learner's initial state
- `A::Matrix{Float64}`: an advantage matrix

As an example, consider:

```@example simple
carla = LRPLearner(2, [0.001, 0.005];
                   β = [0.1, 0.2], 
                   δ = [0.0, 0.3], 
                   W = [0.2, 0.8], 
                   A = [0.0 0.1; 0.4 0.0])
```

Sanity checks are automatically performed to make sure the to-be-created learner is well-defined: for instance, the constructors check that the sum over all elements of `W` equals one, as well as that each element of `W` is non-negative. In case any well-definedness criteria are violated, the constructor returns with an error.


## Actions, rewards and punishments

To choose an action (i.e. to sample from the categorical distribution represented by the current action probabilities i.e. weights `W`), LearningAutomata.jl provides the `act` function. This function returns the index of the action chosen:

```@example simple
act(alice)
```

The `act` function also has a second method which takes a second argument, the number of actions to perform:

```@example simple
act(alice, 10)
```

For low-level control, individual actions can be rewarded and punished using the `reward!` and `punish!` functions. Each function takes two positional arguments, the first one being the learner (e.g. of type `LRPLearner`), the second one being the index of the action to be punished (of type `Int`). For example:

```@example simple
reward!(alice, 1)
punish!(alice, 2)
show(alice)
```

Note how `alice`'s action probability vector has changed in response to the reward and punishment.

As a slightly more complicated example, let's reward a randomly drawn action of `bob`'s 100 times:

```@example simple
[reward!(bob, i) for i in act(bob, 100)]
show(bob)
```


## Simulations in a stationary random environment

As explained in the section on [Theory](@ref), in the normal case we assume the following sort of model:

1. The learner chooses which action to do, drawing action $\mathcal{A}_i$ with probability $W_i$;
2. The environment either rewards or punishes that action;
3. Repeat from 1.

A stationary random environment (SRE) is a vector of penalty probabilities $(c_1, \ldots , c_n)$ where $c_i$ denotes the (constant) probability with which the learner's $i$th action is punished. In LearningAutomata.jl, these environments have type `SRE`.

```@example simple
paris = SRE(0.1, 0.4)
```

To check whether such an environment punishes an action performed on it, LearningAutomata.jl provides the `punishes` function:

```@example simple
punishes(paris, 1)
```

!!! note

    The `punishes` function constitutes a random experiment. Thus, on some calls (here, roughly 10% of them), the function will return `true` for the first action (and it will return `true` on the second action about 40% of the time). Of course, if we repeat this random experiment sufficiently many times, we approach the values of the penalty probabilities:

    ```@example simple
    sum([punishes(paris, 1) for i in 1:1000])/1000
    ```

    ```@example simple
    sum([punishes(paris, 2) for i in 1:1000])/1000
    ```

Assuming such a learning environment, we can easily implement the above "learning loop", say, for 1000 learning iterations. (We set the PRNG seed for reproducibility purposes.)

```@example simple
using Random
Random.seed!(123)

alice = LRPLearner(2, 0.01)
paris = SRE(0.1, 0.3)

for t in 1:1000
  action_chosen = act(alice)
  if punishes(paris, action_chosen)
    punish!(alice, action_chosen)
  else
    reward!(alice, action_chosen)
  end
end

show(alice)
```

Since this sort of use case is common, LearningAutomata.jl provides methods for it. The contents of the above loop are equivalent to the following function call:

```@example simple
interact!(alice, paris)
```

In other words, this carries out one interaction between the learner and its environment (i.e. one action followed by either reward or punishment). To simulate an entire history, an array comprehension can be used:

```@example simple
Random.seed!(123)

alice = LRPLearner(2, 0.01)
paris = SRE(0.1, 0.3)

[interact!(alice, paris) for t in 1:1000]

show(alice)
```

However, one can also use a three-argument version of `interact!`:

```@example simple
Random.seed!(123)

alice = LRPLearner(2, 0.01)
paris = SRE(0.1, 0.3)

history = interact!(alice, paris, 1000)

show(alice)
```

This has the benefit that the learning trajectory is returned as a matrix which can be directly passed onto plotting routines; the matrix has as many columns as there are actions, and the row count equals the length of the learning trajectory:

```@example simple
history
```

!!! tip

    This behaviour can be suppressed by appending the keyword argument `collect_history = false` to `interact!`, in which case the return value is the learner's knowledge state at the end of the simulation, rather than the entire learning trajectory.

The history can be passed into `Plots.plot` to visualize the learning trajectory:

```@example simple
using Plots

plot(history, label=["W1" "W2"]);
ylims!(0.0, 1.0);

savefig("history.png") # hide
nothing # hide
```

![](history.png)


## Interactions between learners

The `interact!` function also provides methods that facilitate the simulation of interactions between two learners (rather than between a learner and an abstract learning environment). By definition, an advantage matrix is a square $n \times n$ matrix $A = [a_{ij}]$ (where $n$ denotes the number of actions) with the following interpretation: $a_{ij}$ is the probability with which action $\mathcal{A}_j$ punishes action $\mathcal{A}_i$.

For purposes of illustration, let us assume the following advantage matrix:

```@example simple
A = [0.0 0.1
     0.2 0.0]
```

This means that the probability of $\mathcal{A}_1$ punishing $\mathcal{A}_2$ is 0.2, while the probability of $\mathcal{A}_2$ punishing $\mathcal{A}_1$ is only 0.1. Furthermore, neither action ever punishes itself (this is often a reasonable assumption in applications but is by no means necessary).

Let us create a couple of learners, one of whom initially prefers action $\mathcal{A}_1$, the other preferring $\mathcal{A}_2$:

```@example simple
daisy = LRPLearner(2, 0.01; W = [0.1, 0.9], A = A)
```

```@example simple
eddie = LRPLearner(2, 0.01; W = [0.9, 0.1], A = A)
```

To make the two learners interact, we call `interact!` on them:

```@example simple
interact!(daisy, eddie)
```

This makes the following things happen:

1. `daisy` samples an action to do, based on her current `W`, and imposes this action on `eddie`
1. `eddie` samples an action to do, based on his current `W`, and imposes this action on `daisy`
1. `daisy` updates her knowledge state appropriately
1. `eddie` updates his knowledge state appropriately

!!! note

    Note the order of these events. Both actions are sampled before any updates are done to knowledge states. In other words, the actions are concurrent.

By default, the interaction is reciprocal, in the sense that both learners update their states. If this is not desired, the keyword argument `reciprocal` is to be set to `false`:

```@example simple
interact!(daisy, eddie; reciprocal = false)
```

In this case, only `eddie` updates his state.

To better observe the changes to the learners' knowledge states, let's make them interact a number of times. To facilitate this sort of use case, the `interact!` function provides a second method that takes three positional arguments: the two learners and an `Int`, the number of interactions to carry out. The keyword argument `reciprocal` determines whether the interactions are reciprocal (`true` by default). By default, the trajectories of both learners are returned as a vector of matrices. This can be suppressed by setting `collect_history = false`.

```@example simple
histories = interact!(daisy, eddie, 1500)

using Plots

plot(histories[1], label=["daisy W1" "daisy W2"])
plot!(histories[2], label=["eddie W1" "eddie W2"])

savefig("interact_trajectory.png") # hide
nothing # hide
```

![](interact_trajectory.png)

More complicated interaction dynamics are best implemented using the tools provided by [Agents.jl](https://juliadynamics.github.io/Agents.jl/stable/), as outlined in the following section.


