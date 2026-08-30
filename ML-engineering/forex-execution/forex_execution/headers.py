from forex_execution.config import oanda_practice_config


def get_oanda_headers() -> dict:
    return {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + oanda_practice_config.OANDA_PRACTICE_TOKEN,
        'Accept-Datetime-Format': 'unix',
    }
