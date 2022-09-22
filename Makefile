### Make File fo CCU_validation_scoring

check:
	pytest

check-scoring:
	pytest -k test/test_scoring.py

check-unittest:
	pytest -k test/test_unittests.py

