"""Silver file names used by the intermediate pipeline (new silver structure).

key (used in the code) -> list of silver files (without extension) in INPUT_DIR.
Several files are concatenated; an empty list means the table does not exist in silver.
"""
SOURCE_TABLES = {
    # Roughness
    "roughness_as_per_ca": ["child_roughness_as_per_ca"],  # CA, at the CA reporting interval
    "roughness_child_100m": ["child_100m_roughness"],      # IRC, 100 m
    "roughness": ["child_10m_roughness"],                  # 10 m, CA & IRC (fallback)
    # Rutting
    "rutting_as_per_ca": ["child_rutting_as_per_ca"],
    "rutting_child_100m": ["child_100m_rutting"],
    "rutting": ["child_10m_rutting"],
    # Cracking
    "cracking_as_per_ca": ["master_cracking_ca"],          # CA, 100 m
    "cracking_child_100m": ["master_cracking_irc"],        # IRC, 100 m
    "cracking": ["child_10m_cracking"],                    # 10 m, CA & IRC (fallback)
}
