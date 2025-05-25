from dynaconf import Dynaconf

settings = Dynaconf(
    envvar_prefix="RECO_TRAIN",
    settings_file="default.toml",
)
