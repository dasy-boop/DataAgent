import pytest

from tools.registry import get_tool


def test_get_registered_tool():
    tool = get_tool("count_values")

    assert tool.name == "count_values"
    assert callable(tool.func)


def test_get_unknown_tool_raises_error():
    with pytest.raises(KeyError):
        get_tool("unknown_tool")