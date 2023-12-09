# Testing command line interfaces is hard. But we'll try
# At least we separated our actual program from the I/O part so that we
# can test that
import wrfvis
from wrfvis.cltools import gridcell


def test_help(capsys):

    # Check that with empty arguments we return the help
    gridcell([])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out
    print(captured.out)

    gridcell(['-h'])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out

    gridcell(['--help'])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out


def test_version(capsys):

    gridcell(['-v'])
    captured = capsys.readouterr()
    assert wrfvis.__version__ in captured.out


def test_print_html(capsys):

    gridcell(['-p', 'T', '-l', '12.1', '47.3', '300', '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out


def test_error(capsys):

    gridcell(['-p', '12.1'])
    captured = capsys.readouterr()
    assert 'command not understood' in captured.out
