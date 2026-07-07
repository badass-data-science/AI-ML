import forex_strategy


def test_package_exposes_a_string_version():
    assert isinstance(forex_strategy.__version__, str)
    assert forex_strategy.__version__
