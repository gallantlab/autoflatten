"""Configuration dataclasses for surface flattening.

This module defines the configuration structures for the pyflatten algorithm.
"""

from dataclasses import dataclass, field
from typing import Optional
import json


@dataclass
class KRingConfig:
    """K-ring distance computation parameters.

    Attributes
    ----------
    k_ring : int
        Number of neighborhood rings (1 through k-hop neighbors).
    n_neighbors_per_ring : int or None
        Number of angular samples per ring.
        Use None to include all neighbors (no angular sampling).
    """

    k_ring: int = 7
    n_neighbors_per_ring: Optional[int] = 12


@dataclass
class ConvergenceConfig:
    """FreeSurfer-style convergence parameters.

    Based on FreeSurfer v6.0.0 mrisurf.c:117-118.

    Attributes
    ----------
    base_tol : float
        Base convergence tolerance (scaled by sqrt((n_avg+1)/1024)).
    max_small : int
        Max consecutive small steps before stopping at a level.
    total_small : int
        Max total small steps across all iterations.
    """

    base_tol: float = 0.2
    max_small: int = 50000
    total_small: int = 15000


@dataclass
class LineSearchConfig:
    """Vectorized line search parameters.

    Attributes
    ----------
    n_coarse_steps : int
        Number of log-spaced step sizes to evaluate.
    max_mm : float
        Maximum displacement in coordinate units.
    min_mm : float
        Minimum displacement in coordinate units.
    """

    n_coarse_steps: int = 15
    max_mm: float = 1000.0
    min_mm: float = 0.001


@dataclass
class PhaseConfig:
    """Configuration for a single optimization phase.

    Uses FreeSurfer-style direct energy weights (l_nlarea, l_dist) instead
    of a ratio. Reference: mrisurf.c mrisIntegrationEpoch().

    Attributes
    ----------
    name : str
        Human-readable phase name.
    l_nlarea : float
        Weight for nonlinear area energy (prevents flipped triangles).
    l_dist : float
        Weight for distance energy (preserves geodesic distances).
    smoothing_schedule : list of int
        List of gradient averaging counts.
    iters_per_level : int
        Maximum iterations per smoothing level.
    base_tol : float or None
        Override base tolerance for this phase (None = use global).
    enabled : bool
        Whether to run this phase.
    """

    name: str
    l_nlarea: float = 1.0
    l_dist: float = 1.0
    smoothing_schedule: list[int] = field(
        default_factory=lambda: [1024, 256, 64, 16, 4, 1, 0]
    )
    iters_per_level: int = 40  # FreeSurfer default (was 200)
    base_tol: Optional[float] = None
    enabled: bool = True


@dataclass
class NegativeAreaRemovalConfig:
    """Configuration for the negative area removal phase.

    Uses FreeSurfer-style fixed l_nlarea=1.0 with varying l_dist ratios.
    Reference: mrisurf.c:8338-8506 (mrisRemoveNegativeArea)

    FreeSurfer always runs through the ENTIRE l_dist_ratios schedule exactly once.
    The early stopping code in FreeSurfer is disabled (#if 0), so all ratio
    passes execute regardless of the current negative area percentage.

    Attributes
    ----------
    base_averages : int
        Starting smoothing level.
    min_area_pct : float
        Target percentage of flipped triangles. Only checked BEFORE starting
        NAR (to skip entirely if already below threshold). Once started,
        all ratio passes run to completion.
    l_nlarea : float
        Fixed nonlinear area weight.
    l_dist_ratios : list of float
        Sequence of distance weights to iterate through.
        FreeSurfer uses [1e-6, 1e-5, 1e-3, 1e-2, 1e-1].
        ALL ratios are always used (no truncation).
    iters_per_level : int
        Maximum iterations per smoothing level.
    base_tol : float
        Convergence tolerance for this phase.
    enabled : bool
        Whether to run negative area removal.
    scale_area : bool
        Whether to apply area-preserving scaling at each iteration.
        This maintains the original 3D surface area during optimization.
        FreeSurfer has this step commented out, so it's disabled by default.
    """

    base_averages: int = 1024  # FreeSurfer default
    min_area_pct: float = 0.5  # Only checked before starting, not during
    l_nlarea: float = 1.0  # Fixed area weight
    # All ratios are used in order (FreeSurfer runs all 5, no truncation)
    l_dist_ratios: list[float] = field(
        default_factory=lambda: [1e-6, 1e-5, 1e-3, 1e-2, 1e-1]
    )
    iters_per_level: int = 30  # FreeSurfer default
    base_tol: float = 0.5
    enabled: bool = True
    scale_area: bool = False


@dataclass
class SpringSmoothingConfig:
    """Configuration for final spring smoothing phase.

    FreeSurfer-style Laplacian smoothing that regularizes triangle shapes,
    producing visually smoother flatmaps at the cost of slightly higher
    distance error.

    Reference: FreeSurfer mrisurf.c:7904-7928

    Attributes
    ----------
    n_iterations : int
        Number of smoothing iterations.
    dt : float
        Step size (l_spring coefficient).
    max_step_mm : float
        Maximum step size per vertex (FreeSurfer's MAX_MOMENTUM_MM).
    enabled : bool
        Whether to run final spring smoothing.
    """

    n_iterations: int = 5
    dt: float = 0.5
    max_step_mm: float = 1.0
    enabled: bool = True


@dataclass
class FinalNegativeAreaRemovalConfig:
    """Configuration for the final negative area removal step.

    This runs after the main optimization epochs with tighter tolerance
    and capped averaging to clean up any remaining flipped triangles.

    Reference: mrisurf.c:7886-7901

    Like the initial NAR, this runs through the full l_dist_ratios schedule.

    Attributes
    ----------
    enabled : bool
        Whether to run final negative area removal.
    base_averages : int
        Starting smoothing level (capped at 32 in FreeSurfer).
    l_nlarea : float
        Fixed nonlinear area weight.
    l_dist_ratios : list of float
        Sequence of distance weights to iterate through.
        Same as initial NAR: [1e-6, 1e-5, 1e-3, 1e-2, 1e-1].
    base_tol : float
        Convergence tolerance (tighter than initial NAR).
    iters_per_level : int
        Maximum iterations per smoothing level.
    """

    enabled: bool = True
    base_averages: int = 32  # Capped at 32 in FreeSurfer
    l_nlarea: float = 1.0
    # Same ratio schedule as initial NAR
    l_dist_ratios: list[float] = field(
        default_factory=lambda: [1e-6, 1e-5, 1e-3, 1e-2, 1e-1]
    )
    base_tol: float = 0.01  # Tighter than initial 0.5
    iters_per_level: int = 30


def _default_phases() -> list[PhaseConfig]:
    """Return default optimization phases matching FreeSurfer's 3 epochs.

    FreeSurfer uses three mrisIntegrationEpoch calls with:
    - Epoch 1: l_nlarea=1.0, l_dist=0.1 (area dominant)
    - Epoch 2: l_nlarea=1.0, l_dist=1.0 (balanced)
    - Epoch 3: l_nlarea=0.1, l_dist=1.0 (distance dominant)

    Reference: mrisurf.c:7805-7839, area_coefs=[1.0, 1.0, 0.1], dist_coefs=[0.1, 1.0, 1.0]
    """
    return [
        PhaseConfig(
            name="epoch_1",
            l_nlarea=1.0,
            l_dist=0.1,
            smoothing_schedule=[1024, 256, 64, 16, 4, 1, 0],
            iters_per_level=40,
        ),
        PhaseConfig(
            name="epoch_2",
            l_nlarea=1.0,
            l_dist=1.0,
            smoothing_schedule=[1024, 256, 64, 16, 4, 1, 0],
            iters_per_level=40,
        ),
        PhaseConfig(
            name="epoch_3",
            l_nlarea=0.1,
            l_dist=1.0,
            smoothing_schedule=[1024, 256, 64, 16, 4, 1, 0],
            iters_per_level=40,
        ),
    ]


@dataclass
class FlattenConfig:
    """Complete configuration for surface flattening.

    Attributes
    ----------
    kring : KRingConfig
        K-ring distance computation parameters.
    convergence : ConvergenceConfig
        Convergence parameters.
    line_search : LineSearchConfig
        Line search parameters.
    negative_area_removal : NegativeAreaRemovalConfig
        Negative area removal phase config.
    final_negative_area_removal : FinalNegativeAreaRemovalConfig
        Final negative area removal config (runs after main epochs).
    spring_smoothing : SpringSmoothingConfig
        Final spring smoothing phase config.
    phases : list of PhaseConfig
        List of optimization phase configurations.
    print_every : int
        Print progress every N iterations.
    verbose : bool
        Whether to print progress messages.
    n_jobs : int
        Number of parallel jobs for distance computation.
        -1 means use all available CPUs.
    strict_topology : bool
        If True, raise error for non-disk topology (chi != 1).
        If False, warn but continue (flattening will likely fail).
    adaptive_recovery : bool
        Enable adaptive flipped-triangle recovery during
        distance refinement phase. When flipped count exceeds threshold,
        temporarily increases area weight to fix flipped triangles.
        Disabled by default to match FreeSurfer's fixed schedule.
    """

    kring: KRingConfig = field(default_factory=KRingConfig)
    convergence: ConvergenceConfig = field(default_factory=ConvergenceConfig)
    line_search: LineSearchConfig = field(default_factory=LineSearchConfig)
    negative_area_removal: NegativeAreaRemovalConfig = field(
        default_factory=NegativeAreaRemovalConfig
    )
    final_negative_area_removal: FinalNegativeAreaRemovalConfig = field(
        default_factory=FinalNegativeAreaRemovalConfig
    )
    spring_smoothing: SpringSmoothingConfig = field(
        default_factory=SpringSmoothingConfig
    )
    phases: list[PhaseConfig] = field(default_factory=_default_phases)
    print_every: int = 100
    verbose: bool = True
    n_jobs: int = -1
    strict_topology: bool = True
    adaptive_recovery: bool = False  # Disabled by default for FreeSurfer mode
    initial_scale: float = 3.0  # Scale factor after initial 2D projection

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        return {
            "kring": {
                "k_ring": self.kring.k_ring,
                "n_neighbors_per_ring": self.kring.n_neighbors_per_ring,
            },
            "convergence": {
                "base_tol": self.convergence.base_tol,
                "max_small": self.convergence.max_small,
                "total_small": self.convergence.total_small,
            },
            "line_search": {
                "n_coarse_steps": self.line_search.n_coarse_steps,
                "max_mm": self.line_search.max_mm,
                "min_mm": self.line_search.min_mm,
            },
            "negative_area_removal": {
                "base_averages": self.negative_area_removal.base_averages,
                "min_area_pct": self.negative_area_removal.min_area_pct,
                "l_nlarea": self.negative_area_removal.l_nlarea,
                "l_dist_ratios": self.negative_area_removal.l_dist_ratios,
                "iters_per_level": self.negative_area_removal.iters_per_level,
                "base_tol": self.negative_area_removal.base_tol,
                "enabled": self.negative_area_removal.enabled,
                "scale_area": self.negative_area_removal.scale_area,
            },
            "final_negative_area_removal": {
                "enabled": self.final_negative_area_removal.enabled,
                "base_averages": self.final_negative_area_removal.base_averages,
                "l_nlarea": self.final_negative_area_removal.l_nlarea,
                "l_dist_ratios": self.final_negative_area_removal.l_dist_ratios,
                "base_tol": self.final_negative_area_removal.base_tol,
                "iters_per_level": self.final_negative_area_removal.iters_per_level,
            },
            "spring_smoothing": {
                "n_iterations": self.spring_smoothing.n_iterations,
                "dt": self.spring_smoothing.dt,
                "max_step_mm": self.spring_smoothing.max_step_mm,
                "enabled": self.spring_smoothing.enabled,
            },
            "phases": [
                {
                    "name": p.name,
                    "l_nlarea": p.l_nlarea,
                    "l_dist": p.l_dist,
                    "smoothing_schedule": p.smoothing_schedule,
                    "iters_per_level": p.iters_per_level,
                    "base_tol": p.base_tol,
                    "enabled": p.enabled,
                }
                for p in self.phases
            ],
            "print_every": self.print_every,
            "verbose": self.verbose,
            "n_jobs": self.n_jobs,
            "strict_topology": self.strict_topology,
            "adaptive_recovery": self.adaptive_recovery,
            "initial_scale": self.initial_scale,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize config to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict) -> "FlattenConfig":
        """Create config from dictionary."""
        kring = KRingConfig(**data.get("kring", {}))
        convergence = ConvergenceConfig(**data.get("convergence", {}))
        line_search = LineSearchConfig(**data.get("line_search", {}))
        negative_area_removal = NegativeAreaRemovalConfig(
            **data.get("negative_area_removal", {})
        )
        final_negative_area_removal = FinalNegativeAreaRemovalConfig(
            **data.get("final_negative_area_removal", {})
        )
        spring_smoothing = SpringSmoothingConfig(**data.get("spring_smoothing", {}))
        phases_data = data.get("phases", _default_phases())
        phases = [
            p if isinstance(p, PhaseConfig) else PhaseConfig(**p) for p in phases_data
        ]
        return cls(
            kring=kring,
            convergence=convergence,
            line_search=line_search,
            negative_area_removal=negative_area_removal,
            final_negative_area_removal=final_negative_area_removal,
            spring_smoothing=spring_smoothing,
            phases=phases,
            print_every=data.get("print_every", 100),
            verbose=data.get("verbose", True),
            n_jobs=data.get("n_jobs", -1),
            strict_topology=data.get("strict_topology", True),
            adaptive_recovery=data.get("adaptive_recovery", False),  # Default to False
            initial_scale=data.get("initial_scale", 3.0),
        )

    @classmethod
    def from_json(cls, json_str: str) -> "FlattenConfig":
        """Create config from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def from_json_file(cls, path: str) -> "FlattenConfig":
        """Load config from JSON file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())


def get_kring_cache_filename(output_path: str, kring_config: KRingConfig) -> str:
    """Generate k-ring cache filename with parameters encoded.

    Parameters
    ----------
    output_path : str
        Path to output file (e.g., "output.patch.3d")
    kring_config : KRingConfig
        K-ring configuration

    Returns
    -------
    str
        Cache filename (e.g., "output.patch.3d.kring_k20_n20.npz")
    """
    if kring_config.n_neighbors_per_ring is None:
        suffix = f"kring_k{kring_config.k_ring}_nall.npz"
    else:
        suffix = (
            f"kring_k{kring_config.k_ring}_n{kring_config.n_neighbors_per_ring}.npz"
        )
    return f"{output_path}.{suffix}"
