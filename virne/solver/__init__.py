import itertools
import warnings
from .base_solver import Solver, SolverRegistry
from .exact import *
from .heuristic import *
from .meta_heuristic import *
try:
    from .learning import *
    from . import learning
except Exception as e:
    learning = None
    warnings.warn(
        f'Learning solvers are unavailable because optional dependencies failed to load: {e}',
        RuntimeWarning,
    )

from . import exact, heuristic, meta_heuristic


SOLVERS = {
    'exact': tuple(exact.__all__),
    'heuristic': tuple(heuristic.__all__),
    'meta_heuristic': tuple(meta_heuristic.__all__),
    'learning': tuple(learning.__all__) if learning is not None else (),
}
SOLVERS['all'] = tuple(itertools.chain.from_iterable(SOLVERS.values()))

__all__ = [
    *SOLVERS['all'],
    'Solver',
    'SolverRegistry',
]