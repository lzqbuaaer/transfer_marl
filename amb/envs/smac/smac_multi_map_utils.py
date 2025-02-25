import enum
from .smac_maps import get_map_params

    
class UnitType(enum.IntEnum):
    MARINE = 0
    ZEALOT = 1
    STALKER = 2
    COLOSSUS = 3
    MARAUDER = 4
    MEDIVAC = 5
    HYDRALISK = 6
    BANEL = 7
    ZERG = 8
    SCV = 9


map_unit_types_universe_registry = {
    "3m": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([48]),
    },
    "5m_vs_6m": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([48]),
    },
    "6m": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([48]),
    },
    "8m": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([48]),
    },
    "3s_vs_3z": {
        'ally': set([UnitType.STALKER]),
        'enemy': set([73]),
    },
    "3s_vs_4z": {
        'ally': set([UnitType.STALKER]),
        'enemy': set([73]),
    },
    "3s_vs_5z": {
        'ally': set([UnitType.STALKER]),
        'enemy': set([73]),
    },
    "2s3z": {
        'ally': set([UnitType.STALKER, UnitType.ZEALOT]),
        'enemy': set([73, 74]),
    },
    "2m_vs_1z": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([73]),
    },
    
    # Dual
    "3m_dual": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([UnitType.MARINE]),
    },
    "5m_vs_6m_dual": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([UnitType.MARINE]),
    },
    "6m_dual": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([UnitType.MARINE]),
    },
    "8m_dual": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([UnitType.MARINE]),
    },
    "3s_vs_3z_dual": {
        'ally': set([UnitType.STALKER]),
        'enemy': set([UnitType.ZEALOT]),
    },
    "3s_vs_4z_dual": {
        'ally': set([UnitType.STALKER]),
        'enemy': set([UnitType.ZEALOT]),
    },
    "3s_vs_5z_dual": {
        'ally': set([UnitType.STALKER]),
        'enemy': set([UnitType.ZEALOT]),
    },
    "2s3z_dual": {
        'ally': set([UnitType.STALKER, UnitType.ZEALOT]),
        'enemy': set([UnitType.STALKER, UnitType.ZEALOT]),
    },
    "2m_vs_1z_dual": {
        'ally': set([UnitType.MARINE]),
        'enemy': set([UnitType.ZEALOT]),
    },
}

def get_alignment_shield_bits(map_name_list, host):
    shield_bits_ally_alignment, shield_bits_enemy_alignment = 0, 0
    for map_name in map_name_list:
        map_params = get_map_params(map_name)
        if host:
            shield_bits_ally_alignment = shield_bits_ally_alignment or (map_params["a_race"] == 'P')
            shield_bits_enemy_alignment = shield_bits_enemy_alignment or (map_params["b_race"] == 'P')
        else:
            shield_bits_ally_alignment = shield_bits_ally_alignment or (map_params["b_race"] == 'P')
            shield_bits_enemy_alignment = shield_bits_enemy_alignment or (map_params["a_race"] == 'P')
    return shield_bits_ally_alignment, shield_bits_enemy_alignment

def get_unit_types_universe(map_name_list, host):
    ally_universe, enemy_universe = set([]), set([])
    for map_name in map_name_list:
        if host:
            ally_universe = ally_universe | map_unit_types_universe_registry[map_name]['ally']
            enemy_universe = enemy_universe | map_unit_types_universe_registry[map_name]['enemy']
        else:
            ally_universe = ally_universe | map_unit_types_universe_registry[map_name]['enemy']
            enemy_universe = enemy_universe | map_unit_types_universe_registry[map_name]['ally']
    return ally_universe, enemy_universe