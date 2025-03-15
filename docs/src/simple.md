# Learning in simple environments

Here, we illustrate the use of LearningAutomata.jl in simple instances in which the learning environment is stationary and the learner employs the linear reward--penalty (LRP) scheme (see the [Theory](@ref) section for details).


## The `LRPLearner` type and its constructors

The central structure provided by the package is the `LRPLearner` type, representing a learner that updates its knowledge state using the LRP scheme. Two constructors are provided for this type. The first one takes two positional arguments, one `Int` and one `Float64`:

```@repl
using LearningAutomata
alice = LRPLearner(2, 0.001)
```

The first argument sets the number of actions, the second argument sets a global learning rate that applies to all actions and both rewards and penalties (see [Theory](@ref) for explanation). The learner's initial state is automatically set to the maximum-entropy state $(1/n, \ldots , 1/n)$, where $n$ is the number of actions.

The second constructor takes, like the first one, the number of actions as the first positional argument. The second positional argument, however, is of type `Vector{Float64}`. This makes it possible to give each action its own learning rate, e.g.:

```@repl
using LearningAutomata # hide
bob = LRPLearner(3, [0.001, 0.005, 0.02])
```

Each constructor furthermore accepts a number of keyword arguments to further specify the learner's behaviour. These are:

- `thetaP::Vector{Float64}`: a learning rate vector that applies to punishments (the $i$th component of this vector applying to the $i$th action)
- `delta::Vector{Float64}`: a vector of action costs
- `W::Vector{Float64}`: the learner's initial state
- `A::Matrix{Float64}`: an advantage matrix (see [Theory](@ref) for discussion and REF for applications of this concept)

As an example, consider:

```@repl
using LearningAutomata # hide
carla = LRPLearner(2, 
                   [0.001, 0.005];
                   b = [0.1, 0.2], 
                   c = [0.0, 0.3], 
                   W = [0.2, 0.8], 
                   A = [0.0 0.1; 0.4 0.0])
```

Sanity checks are automatically performed to make sure the to-be-created learner is well-defined: for instance, the constructors check that the sum over all elements of `W` equals one, as well as that each element of `W` is non-negative. In case any well-definedness criteria are violated, the constructor returns with an error.


## Rewards and punishments

For low-level control, individual actions can be rewarded and punished using the `reward!` and `punish!` functions. Each function takes two positional arguments, the first one being the learner (e.g. of type `LRPLearner`), the second one being the index of the action to be punished (of type `Int`). For example:

```@repl
using LearningAutomata # hide
alice = LRPLearner(2, 0.001) # hide
reward!(alice, 1)
punish!(alice, 2)
show(alice)
```

Note how `alice`'s action probability vector has changed in response to the reward and punishment.

As a slightly more complicated example, let's reward a randomly drawn action of `bob`'s 100 times:

```@repl
using LearningAutomata # hide
bob = LRPLearner(3, [0.001, 0.005, 0.02]) # hide
[reward!(bob, i) for i in rand(1:bob.n, 100)]
show(bob)
```


## Simulations in a stationary random environment

As explained in the section on [Theory](@ref), in the normal case we assume the following sort of model:

1. The learner chooses which action to do, drawing action $\alpha_i$ with probability $W_i$;
2. The environment either rewards or punishes that action;
3. Repeat from 1.

As also explained in [Theory](@ref), a stationary random environment (SRE) is a vector of penalty probabilities $(c_1, \ldots , c_n)$ where $c_i$ denotes the (constant) probability with which the learner's $i$th action is punished.

With this information, we can easily implement the above "learning loop", say, for 100 learning iterations:

```@repl
using LearningAutomata # hide
using Random
using StatsBase

Random.seed!(123)

alice = LRPLearner(2, 0.01)
paris = [0.1, 0.3]

for t in 1:100
  action_chosen = StatsBase.sample(1:alice.n, Weights(alice.W))
  if rand() < paris[action_chosen]
    punish!(alice, action_chosen)
  else
    reward!(alice, action_chosen)
  end
end

show(alice)
```

Since this sort of implementation is common, LearningAutomata.jl provides a convenience function, `simulate!`, to automate it:

```@repl
using LearningAutomata # hide
using Random
using StatsBase

Random.seed!(123)

alice = LRPLearner(2, 0.01)
paris = [0.1, 0.3]

simulate!(alice, 100, paris)

show(alice)
```

The output is FIXME



## Reconfiguring learners


## Games
