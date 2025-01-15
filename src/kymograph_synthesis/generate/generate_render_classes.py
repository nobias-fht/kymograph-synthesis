from ..params.render_params import FluoroDistrName
from ..render.fluorophore_distributions import SimplexNoise

def fluoro_distr_factory(*, name: FluoroDistrName, **kwargs):
    match name:
        case FluoroDistrName.SIMPLEX_NOISE:
            return SimplexNoise(**kwargs)
        case _:
            raise ValueError(
                f"Unrecognised fluorophore distribution name '{name}'."
            )
        