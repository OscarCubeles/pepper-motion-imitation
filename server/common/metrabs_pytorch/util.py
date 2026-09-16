import functools
import os.path as osp

import hydra
import hydra.core.global_hydra

_cfg = None

@functools.lru_cache()
def get_config(config_name=None):
    global _cfg
    if _cfg is not None:
        return _cfg
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    if config_name is not None and osp.isabs(config_name):
        config_path = osp.dirname(config_name)
        config_name = osp.basename(config_name)
        hydra.initialize_config_dir(config_path, version_base='1.1')
    else:
        hydra.initialize(config_path='config', version_base='1.1')

    _cfg = hydra.compose(
        config_name=config_name)
    return _cfg
