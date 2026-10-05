from faultlines.toolguard import ToolGuard, ToolRegistry, render_call
from faultlines.toolguard.gemma_syntax import CALL_OPEN, STR

REG = ToolRegistry.default_harness()


def check(text, **kw):
    return ToolGuard(REG, detect_loops=False, **kw).check(text)


def codes(r):
    return {i.code.value for i in r.issues}


def test_valid_call():
    r = check(render_call("read_file", {"filepath": "a.py", "start_line": 3}))
    assert r.status == "valid" and r.calls[0].args["start_line"] == 3


def test_alias_repair():
    r = check(render_call("bash", {"cmd": "ls"}))
    assert r.status == "repaired" and r.calls[0].name == "run_command" and r.calls[0].args == {"command": "ls"}
    assert "UNKNOWN_TOOL" in codes(r)


def test_truncated_content_is_refused_not_invented():
    r = check(f"{CALL_OPEN}call:write_file{{path:{STR}a.py{STR},content:{STR}def f(")
    assert r.status == "rejected" and codes(r) == {"TRUNCATED_CALL"}
    assert "cut off" in r.feedback


def test_missing_close_tag_with_complete_args_is_repaired():
    r = check(render_call("read_file", {"filepath": "a.py"})[:-len("<tool_call|>")])
    assert r.status == "repaired" and "TRUNCATED_CALL" in codes(r)


def test_json_args_are_parsed_as_json_not_bare_strings():
    r = check('<|tool_call>call:read_file{"filepath": "a.py", "start_line": "5"}<tool_call|>')
    assert r.status == "repaired"
    assert r.calls[0].args == {"filepath": "a.py", "start_line": 5}


def test_extra_arg_needs_lossy():
    text = render_call("read_file", {"filepath": "a.py", "verbose": True})
    assert check(text).status == "rejected"
    r = check(text, allow_lossy=True)
    assert r.status == "repaired" and r.to_dict()["lossy"] is True


def test_no_call_and_path_escape():
    assert check("just thinking").status == "no_call"
    r = check(render_call("read_file", {"filepath": "../../etc/passwd"}))
    assert r.status == "rejected" and "PATH_ESCAPE" in codes(r)


def test_structured_calls():
    g = ToolGuard(REG, detect_loops=False)
    assert g.check_structured("read_file", '{"filepath": "x"}').status == "valid"
    assert g.check_structured("read_file", "{'path': 'x',}").status == "repaired"
    assert g.check_structured("read_file", "not json").status == "rejected"


def test_loop_detection():
    g = ToolGuard(REG)
    text = render_call("read_file", {"filepath": "a.py"})
    results = [g.check(text) for _ in range(3)]
    assert results[-1].loop is not None and "REPEATED_CALL" in codes(results[-1])
