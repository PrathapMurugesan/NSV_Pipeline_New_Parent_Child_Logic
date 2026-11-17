"""Silver defect tables read by RIS View (new silver structure).

silver file (without extension) -> (Defect_Type written to the output, standard kept from that file)
standard None = keep the file's rows as they are ('As per CA' rows count for CA, 'As per IRC' rows for IRC).

Defect counts (new KPI logic):
  Roughness / Rutting : CA  = records of child_<d>_as_per_ca (CA reporting interval)
                        IRC = records of child_100m_<d> (100 m)
  Cracking            : CA  = master_cracking_ca (100 m) | IRC = master_cracking_irc (100 m)
  Pothole             : defect records of master_pothole (CA rows -> CA count, IRC rows -> IRC count)
Ravelling is not part of the silver layer any more.
"""
CA, IRC = "As per CA", "As per IRC"

DEFECT_TABLES = {
    "child_roughness_as_per_ca": ("Roughness", CA),
    "child_100m_roughness": ("Roughness", IRC),
    "child_rutting_as_per_ca": ("Rutting", CA),
    "child_100m_rutting": ("Rutting", IRC),
    "master_cracking_ca": ("Cracking", CA),
    "master_cracking_irc": ("Cracking", IRC),
    "master_pothole": ("Pothole", None),
}
