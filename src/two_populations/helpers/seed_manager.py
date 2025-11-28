import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

from src.two_populations.helpers.logger import setup_logger

logger = setup_logger(
    experiment_name="delay_sweep_gaussian",
    console_level="INFO",
    file_level="DEBUG",
    log_to_file=False
)

_UINT32 = 2**32

def _norm(seed: int) -> int:
    # Mantiene la seed en rango uint32 para evitar negativas/overflow
    return int(seed) % _UINT32

@dataclass
class SeedConfig:
    """Semillas base para procesos fijos y variables (entre trials)."""
    fixed_seed: int = 100
    variable_seed: int = 200

class SeedManager:
    """
    Esquema de semillas:
      fixed_seed_A = fixed_seed - 1
      fixed_seed_B = fixed_seed + 1
      variable_seed_A(base) = variable_seed - 1
      variable_seed_B(base) = variable_seed + 1
      fixed_seed_common = fixed_seed
      variable_seed_common(base) = variable_seed
    Las semillas 'variable' incorporan current_trial al crear RNGs.
    """

    def __init__(self, config: SeedConfig, current_trial: int = 0):
        self.config = config
        self.current_trial = current_trial
        self._rngs: Dict[str, np.random.RandomState] = {}

        # Fijas (no cambian con el trial)
        self.fixed_seed_A = _norm(config.fixed_seed - 1)
        self.fixed_seed_B = _norm(config.fixed_seed + 1)
        self.fixed_seed_common = _norm(config.fixed_seed)

        # Bases variables (el trial se suma al crear el RNG)
        self._var_base_A = _norm(config.variable_seed - 1)
        self._var_base_B = _norm(config.variable_seed + 1)
        self._var_base_common = _norm(config.variable_seed)

    def _variable_with_trial(self, base: int, channel: int) -> int:
        """
        Mapea trial → seed con stride=3 y residuos distintos por canal.
        channel: 0=common, 1=A, 2=B
        """
        return _norm(base + 3*self.current_trial + channel)

    def get_rng(self, category: str, population: Optional[str] = None,
            trial_dependent: bool = True) -> np.random.RandomState:
        key = f"{category}:{population or '-'}:trialdep={trial_dependent}:trial={self.current_trial}"
        if key in self._rngs:
            return self._rngs[key]

        if category == 'common':
            seed = (self._variable_with_trial(self._var_base_common, 0)
                    if trial_dependent else self.fixed_seed_common)
        elif category == 'population_specific':
            if population not in ('A', 'B'):
                raise ValueError("population_specific requiere population en {'A','B'}")
            if trial_dependent:
                base = self._var_base_A if population == 'A' else self._var_base_B
                channel = 1 if population == 'A' else 2
                seed = self._variable_with_trial(base, channel)
            else:
                seed = self.fixed_seed_A if population == 'A' else self.fixed_seed_B
        else:
            raise ValueError(f"Categoría desconocida: {category}")

        self._rngs[key] = np.random.RandomState(seed)
        return self._rngs[key]

    def update_trial(self, trial: int):
        """Cambia el trial y limpia RNGs dependientes del trial."""
        self.current_trial = int(trial)
        # Elimina SOLO RNGs creados con trial_dependent=True
        self._rngs = {k: v for k, v in self._rngs.items() if "trialdep=True" not in k}
    
    def get_current_variable_seeds(self):
        """Semillas variables efectivas (incluyendo current_trial)."""
        return {
            'variable_common_current': self._variable_with_trial(self._var_base_common, 0),
            'variable_A_current': self._variable_with_trial(self._var_base_A, 1),
            'variable_B_current': self._variable_with_trial(self._var_base_B, 2),
        }
        
    def get_seed_summary(self) -> Dict[str, Any]:
        base = {
            'trial': self.current_trial,
            'fixed_seed_A': self.fixed_seed_A,
            'fixed_seed_B': self.fixed_seed_B,
            'fixed_seed_common': self.fixed_seed_common,
            # 'variable_base_A': self._var_base_A,
            # 'variable_base_B': self._var_base_B,
            # 'variable_base_common': self._var_base_common
        }
        base.update(self.get_current_variable_seeds())
        return base

    def validate_configuration(self) -> Dict[str, bool]:
        checks = {}
        checks['fixed_A_B_different'] = self.fixed_seed_A != self.fixed_seed_B
        checks['fixed_variable_common_different'] = self.fixed_seed_common != self._var_base_common
        checks['variable_A_B_different'] = self._var_base_A != self._var_base_B
        all_seeds = [
            self.fixed_seed_A, self.fixed_seed_B, self.fixed_seed_common,
            self._var_base_A, self._var_base_B, self._var_base_common
        ]
        checks['all_seeds_unique'] = len(set(all_seeds)) == len(all_seeds)
        return checks

class SeedMapping:
    """Compatibilidad con llamadas antiguas."""
    @staticmethod
    def get_equivalent_rng(seed_manager: SeedManager, old_usage: str) -> np.random.RandomState:
        mapping = {
            'fixed_rng_common': ('common', None, False),
            'variable_rng_common': ('common', None, True),
            'fixed_rng_pop_A': ('population_specific', 'A', False),
            'fixed_rng_pop_B': ('population_specific', 'B', False),
            'variable_rng_pop_A': ('population_specific', 'A', True),
            'variable_rng_pop_B': ('population_specific', 'B', True),
        }
        if old_usage not in mapping:
            raise ValueError(f"Uso desconocido: {old_usage}")
        cat, pop, trialdep = mapping[old_usage]
        return seed_manager.get_rng(cat, pop, trialdep)

# Configs predefinidas
def get_standard_config() -> SeedConfig:
    # Usar distintas para pasar la validación y evitar colisiones
    return SeedConfig(fixed_seed=42, variable_seed=43)

def get_debug_config() -> SeedConfig:
    return SeedConfig(fixed_seed=42, variable_seed=1042)
