import dd_agent


def test_package_imports():
    assert dd_agent is not None


def test_package_has_version():
    assert isinstance(dd_agent.__version__, str)
    assert len(dd_agent.__version__) > 0
