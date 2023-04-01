default_pysr_kwargs = dict(
    populations=20,         # how many equations are in one population
    procs=20,               # how many processes are done in parallel
    progress=False,
    verbosity=0,
)

default_syreal_kwargs = dict(
    upper_sigma=0.0,
    niterations=30,
    unary_operators=["neg", "square", "cube",
                     "exp", "sqrt", "sin", "cos", "tanh"],
    binary_operators=["plus", "sub", "mult", "div"],
    check_if_loss_zero=True,
    N_start=5,
    equation_tracking=False,
    early_stop=True,
    pysr_params=default_pysr_kwargs,
    abs_loss_zero_tol=1e-6,
)
