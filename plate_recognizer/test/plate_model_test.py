import pytest


from importlib.util import find_spec
if find_spec("plate_recognizer") is None:
    import sys
    sys.path.append('..')

from models.plate_model import PlateModel

def test_plate_model():
    model = PlateModel()
    assert model is not None

if __name__ == '__main__':
    pytest.main()