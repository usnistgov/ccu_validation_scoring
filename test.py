import os, sys, pathlib
import pytest

os.chdir( pathlib.Path.cwd() / 'test' )
pytest.main(sys.argv[1:])
