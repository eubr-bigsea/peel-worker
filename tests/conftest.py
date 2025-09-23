import pytest
from sklearn.datasets import load_iris, load_diabetes, load_breast_cancer


@pytest.fixture
def create_iris():
    return load_iris(as_frame=True)


@pytest.fixture
def create_diabetes():
    return load_diabetes(as_frame=True)

@pytest.fixture
def create_breast():
    return load_breast_cancer(as_frame=True)