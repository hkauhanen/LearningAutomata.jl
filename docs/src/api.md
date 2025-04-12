# API

## Composite types and constructors

### `LRPLearner`

```@docs
LRPLearner
LRPLearner(::Int, ::Float64)
LRPLearner(::Int, ::Vector{Float64})
```

### `LRILearner`

```@docs
LRILearner
LRILearner(::Int, ::Float64)
LRILearner(::Int, ::Vector{Float64})
```

### `SRE`

```@docs
SRE
```


## Methods

### Learner methods

```@docs
act
reset!
reward!
punish!
interact!(x::AbstractLearner, y::AbstractLearner)
interact!(x::AbstractLearner, y::AbstractLearner, n::Int)
```

### Environment methods

```@docs
interact!(x::AbstractLearner, y::AbstractSRE)
interact!(x::AbstractLearner, y::AbstractSRE, n::Int)
limit
limit_pdf
limit_rand
punishes
```

### Auxiliary methods


## Abstract types

### Learners

```@docs
AbstractLearner
AbstractLinearLearner
AbstractLRPLearner
AbstractLRILearner
```

### Learning environments

```@docs
AbstractLearningEnvironment
AbstractSRE
```
