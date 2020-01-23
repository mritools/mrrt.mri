.PHONY : black build clean clean_pyc develop flake8 test

all: develop

flake8:
	flake8 .

clean:
	-python setup.py clean

clean_pyc:
	-find . -name '*.py[co]' -exec rm {} \;

build: clean_pyc
	python setup.py build_ext --inplace

black:
	black .

develop: build
	python -m pip install -e . -v  --no-build-isolation --no-use-pep517

test:
	py.test --pyargs mrrt.mri --cov=mrrt.mri --cov-report term-missing --cov-report html
