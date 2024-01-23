# Testing command line interfaces is hard. But we'll try
# At least we separated our actual program from the I/O part so that we
# can test that
import wrfvis
from wrfvis.cltools import gridcell, skewt


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


def test_help_skewT(capsys):
    '''
    Test the help. It could be integrated in test_help()

    Author
    -------
    Christian Brida
    '''

    # Check that with empty arguments we return the help
    skewt([])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out
    print(captured.out)

    skewt(['-h'])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out

    skewt(['--help'])
    captured = capsys.readouterr()
    assert 'Usage:' in captured.out


def test_version_skewT(capsys):
    '''
    Test the help. It could be integrated in test_version()

    Author
    -------
    Christian Brida
    '''
    skewt(['-v'])
    captured = capsys.readouterr()
    assert wrfvis.__version__ in captured.out


def test_print_html_skewT(capsys):
    '''
    Test the help. It could be integrated in test_print_html()

    Author
    -------
    Christian Brida
    '''

    skewt(['-l', '11', '45', '-t', '2018-08-18T12:00', '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out


def test_print_html_delta_skewT(capsys):
    '''
    Test the help. It could be integrated in test_print_html()

    Author
    -------
    Christian Brida
    '''

    skewt(['-l', '11', '45', '-t', '2018-08-18T12:00', '12', '--no-browser'])
    captured = capsys.readouterr()
    assert 'File successfully generated at:' in captured.out


def test_error_skewT(capsys):

    skewt(['-t', '2018-08-18 12:00', '--no-browser'])
    captured = capsys.readouterr()
    assert 'command not understood' in captured.out

    skewt(['-t', '2018-08-18T12:00', '--no-browser'])
    captured = capsys.readouterr()
    assert 'command not understood' in captured.out
