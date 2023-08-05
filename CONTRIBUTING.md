Developers Guide
---

## Code Organization

In this section, we describe the structure of the codebase. 

### Representations

In saturn/core, we define the following templates & structures. In executors/Technique.py, we provide the abstract skeleton for implementing parallelism techniques. In representations/Strategy.py, we define a data class that allows us to easily aggregate per-model decisions (e.g. how many GPUs, which technique). In representations/Task.py, we provide a data class that lets us collect data on user-submitted models.


### Library
In saturn/library/library.py, we define the basic functions for registering and accessing parallelism techniques. These functions are exposed to the user as part of our API.
We would welcome new contributions to build a "default" library implementing popular parallelisms (e.g. FSDP, GPipe, etc).

### Solver
In ``saturn/solver/milp.py``, we define the MILP for our joint optimization problem of parallelism selection, resource apportioning, and scheduling.
Alternative solvers (e.g. an RL-based one) must follow the same specification (i.e. input/outputs.)

### Trial Runner
In ``saturn/trial_runner/PerformanceEvaluator.py``, we define the empirical profiler system.

### Executor
In ``saturn/executor/executor.py``, we define the execution engine and introspection scheme.

## Tests & Examples
In the examples folder, we provide example training pipelines with Saturn.
If you contribute new parallelisms, you will have to register them with the library before use. You can model your tests
after our example WikiText job.
