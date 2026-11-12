import json

import pytest

from competing_events.cli import main


VALID = {'initial':10,'causes':['A','B'],'rows':[{'time':1,'events':[2,3],'censored':5}],'query_times':[0,1]}


def test_cli_roundtrip_and_readonly(tmp_path,capsys):
    path = tmp_path/'input.json'
    original = json.dumps(VALID)
    path.write_text(original)
    assert main([str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['points'][0]['cumulative_incidence'] == ['1/5','3/10']
    assert result['queries'][0]['survival'] == '1'
    assert path.read_text() == original


@pytest.mark.parametrize('input_text', ['{}','[]','invalid','{"initial":1,"initial":1}',json.dumps({**VALID,'extra':0}),json.dumps({**VALID,'query_times':[2]}),json.dumps({**VALID,'query_times':None}),json.dumps({**VALID,'rows':[{'time':1,'events':[2,3],'censored':5,'extra':1}]}),' '*(1024*1024+1)],ids=['empty','array','syntax','duplicate','unknown','extrapolate','null-query','row-field','oversize'])
def test_invalid_cli(tmp_path,capsys,input_text):
    path = tmp_path/'input.json'
    path.write_text(input_text)
    assert main([str(path)]) == 1
    output = capsys.readouterr()
    assert output.out == ''
    assert output.err.startswith('competing-events:')


def test_missing_file(tmp_path,capsys):
    assert main([str(tmp_path/'missing')]) == 1
    assert capsys.readouterr().err.startswith('competing-events:')
