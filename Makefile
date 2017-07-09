PIP :=pip
PYTHON :=python
NAME :=amii-tf-nn
LIB_NAME :=amii_tf_nn

.PHONY: default
default: install

.PHONY: test
test: setup.py
	$(PYTHON) setup.py -q test --addopts '-x $(ARGS)'

.PHONY: test-cov
test-cov: ARGS=--cov $(LIB_NAME) --cov-report term:skip-covered
test-cov: test

.PHONY: install
install: requirements.txt
	$(PIP) install -r requirements.txt $(ARGS)

.PHONY: mnist
mnist:
	$(PYTHON) exe/amii_tf_nn_mnist_example

print-%:
	@echo $*=$($*)

clean:
	-find . -name "*.pyc" -delete
	-find . -name "__pycache__" -delete
	-find $(LIB_NAME) -name "*.so" -delete
	-rm -rf .cache *.egg .eggs *.egg-info build 2> /dev/null
