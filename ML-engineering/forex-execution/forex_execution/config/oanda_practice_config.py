import os

# Deliberately separate from ETL-forex-time-series-data's forex.oanda.config.oanda_config
# (which only knows OANDA_LIVE_*): this package only ever talks to the practice server,
# and keeping the two configs distinct means a missing/misnamed live credential can
# never accidentally let this tool fall through to a real-money account.
_REQUIRED_KEYS = frozenset(['OANDA_PRACTICE_SERVER', 'OANDA_PRACTICE_TOKEN', 'OANDA_PRACTICE_ACCOUNT_ID'])


def __getattr__(name):
    if name in _REQUIRED_KEYS:
        try:
            return os.environ[name]
        except KeyError:
            raise AttributeError(f'environment variable {name!r} is not set') from None
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
