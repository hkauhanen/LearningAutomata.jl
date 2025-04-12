# Working with mean dynamics

```@setup m
using LearningAutomata
```


## The long-learning limit

Suppose a learner operates in a stationary random environment (SRE), employing the linear reward--penalty (LRP) scheme. As the learning process unfolds, the learner's state $\mathbf{W}$ converges in distribution to a random variable $\mathbf{W}^\infty$ with mean $\mathbf{v}^\infty := \mathbb{E}\mathbf{W}^\infty$. If the SRE is such that $c_i = 0$ is for a unique $i$ (in other words, there is a single action that is never punished, while all the other actions are sometimes punished), when $\mathbf{v}^\infty = \mathbf{e}_i$, where $\mathbf{e}_i$ is the $i$th standard basis vector in $\mathbb{R}^n$. In other words, the action which is never punished ultimately hogs all weight.

If $c_i > 0$ for all $i$, so that each action is punished some of the time, then

```math
v_i^\infty = \frac{c_i^{-1}}{\sum_{j=1}^n c_j^{-1}}.
```

If $c_i = 0$ for more than one $i$, then $\mathbf{v}^\infty$ depends on the distribution of $\mathbf{W}$ at the start of the learning process.

Given an environment of type `SRE`, the `limit` function can be used to obtain the value of $\mathbf{v}^\infty$ in the first two of the above three cases. (In the third case, the function returns with an error.)

```@example m
paris = SRE(0.2, 0.6, 0.9)

limit(paris)
```

```@example m
london = SRE(0.2, 0.0, 0.9)

limit(london)
```

A related function, `limit_rand`, returns a random vector from the vicinity of this mean. The second argument (which must be supplied and has no default value) controls the tightness of the distribution:

```@example m
limit_rand(paris, 1.0)
```

A three-argument method of the function furthermore allows to specify the number of random vectors drawn:

```@example m
limit_rand(paris, 1.0, 10)
```

!!! note

    Technically, `limit_rand` sets up and samples from a Dirichlet distribution with shape parameters FIXME

These methods may be useful in simulations involving very large numbers of learners. Rather than simulating each learner's entire learning trajectory, we may simply sample action probability vectors from the limiting distribution (or an approximation of it, anyway).

To illustrate, consider the following code which simulates 10,000 learners in a SRE and plots the histograms of the action weights at the end (after 1000 learning iterations).

```@example m
lyon = SRE(0.3, 0.6, 0.9)

function my_simulation()
    [interact!(LRPLearner(3, 0.1), lyon, 1000; collect_history = false) for i in 1:10_000]
end

using SimplexPlots

my_simulation() |> simplex_histogram
```

With `limit_rand`, we have the following solution:

```@example m
limit_rand(lyon, 0.2, 10_000) |> simplex_histogram
```

A third option is provided by the fact that the learning automaton is ergodic. In other words, sampling the end state of 10,000 learners leads to the same outcome as sampling 10,000 states from a single learner after the initial transient has died out:

```@example m
function my_ergodic_solution()
    interact!(LRPLearner(3, 0.1), lyon, 11_000; collect_history = true)[1001:end, :]
end

my_ergodic_solution() |> simplex_histogram
```

The agreement of the histograms is acceptable; although the Dirichlet distribution has a slightly different shape, it may be a useful approximation in many cases (of course, the sigma parameter requires tuning).

The three solutions have vastly differing performances:

```@example m
using BenchmarkTools

@benchmark my_simulation()
```

```@example m
@benchmark limit_rand(lyon, 0.1, 10_000)
```

```@example m
@benchmark my_ergodic_solution()
```

The ergodic method is about 1000 times faster than the naive simulation method, and sampling from the Dirichlet distribution is about 2 times faster than the ergodic method.


## The slow-learning limit
