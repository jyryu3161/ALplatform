"""Model names and installation hints, available without numerical imports."""

MODEL_CATALOG = {
    "tabm": {"implementation": "tabm.TabM", "install": "pip install -e .", "access": "local training"},
    "random_forest": {"implementation": "sklearn.ensemble.RandomForestRegressor",
                      "install": "pip install -e .", "access": "local training"},
    "extra_trees": {"implementation": "sklearn.ensemble.ExtraTreesRegressor",
                    "install": "pip install -e .", "access": "local training"},
    "nori": {"implementation": "synthefy_nori.NoriRegressor", "install": "pip install -e '.[nori]'",
             "access": "public pretrained weights; local inference; no API key"},
    "tabldm": {"implementation": "tabldm.TabLDMRegressor", "install": "pip install -e '.[tabldm]'",
               "access": "public pretrained weights; local inference; no API key"},
    "tabfm": {"implementation": "tabfm.TabFMRegressor", "install": "pip install -e '.[tabfm]'",
              "access": "Python >=3.11; public weights for non-commercial, non-production use"},
    "tabpfn": {"implementation": "tabpfn.TabPFNRegressor (TabPFN v2 regression checkpoint)",
               "install": "pip install -e .",
               "access": ("public weights (Prior Labs License: Apache 2.0 + attribution); "
                          "local inference; no API key")},
}

ALIASES = {"tamb": "tabm", "rf": "random_forest", "et": "extra_trees"}


def canonical_model_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("model names must be strings")
    name = ALIASES.get(name.lower(), name.lower())
    if name not in MODEL_CATALOG:
        raise ValueError(f"Unknown model {name!r}; choose from {', '.join(MODEL_CATALOG)}")
    return name
