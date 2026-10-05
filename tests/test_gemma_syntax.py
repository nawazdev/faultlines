import pytest
from faultlines.toolguard.gemma_syntax import (CALL_CLOSE, CALL_OPEN, STR, GemmaSyntaxError, extract_calls,
                                               parse_body, render_call)


def test_round_trip_nested():
    args = {"path": "a, b {c}.py", "n": 3, "x": 1.5, "flag": False, "items": ["a", 2, None], "obj": {"k": "v"}}
    raws = extract_calls("prefix " + render_call("tool", args) + " suffix")
    assert len(raws) == 1 and raws[0].closed
    pb = parse_body(raws[0].body)
    assert pb.name == "tool" and pb.args == args and pb.warnings == []


def test_doc_example():
    text = f'{CALL_OPEN}call:get_current_weather{{location:{STR}Tokyo, JP{STR}}}{CALL_CLOSE}'
    pb = parse_body(extract_calls(text)[0].body)
    assert pb.args == {"location": "Tokyo, JP"}


def test_close_tag_inside_string_is_not_the_end():
    args = {"content": f"literal {CALL_CLOSE} inside"}
    raws = extract_calls(render_call("write_file", args))
    assert raws[0].closed and parse_body(raws[0].body).args == args


def test_truncated_call_is_unclosed():
    raws = extract_calls(f"{CALL_OPEN}call:write_file{{content:{STR}abc")
    assert len(raws) == 1 and not raws[0].closed
    with pytest.raises(GemmaSyntaxError) as e:
        parse_body(raws[0].body)
    assert e.value.code == "UNTERMINATED_STRING"


def test_multiple_calls_and_zero_arg():
    text = render_call("read_file", {"path": "a"}) + render_call("submit_patch", {})
    raws = extract_calls(text)
    assert [parse_body(r.body).name for r in raws] == ["read_file", "submit_patch"]
    assert parse_body("call:submit_patch").args == {}


def test_trailing_text_rejected_unless_allowed():
    with pytest.raises(GemmaSyntaxError):
        parse_body("call:t{a:1} and more")
    assert parse_body("call:t{a:1} and more", allow_trailing=True).warnings == ["TRAILING_TEXT"]
