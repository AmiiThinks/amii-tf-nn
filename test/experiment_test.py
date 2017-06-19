from amii_tf_nn.experiment import Experiment
import os
from yaml import load
my_dir = os.path.dirname(os.path.realpath(__file__))


def test_load_config():
    document = """
name: hi
tag: '2'
seed: 42
root: /
    """
    patient = Experiment.from_yml(document)
    assert patient.name == 'hi'
    assert patient.tag == '2'
    assert patient.seed == 42
    assert patient.root == '/'


def test_load_config_file():
    patient = Experiment.from_yml_file(
        os.path.join(my_dir, 'support', 'experiment_config.yml')
    )
    assert patient.name == 'hi'
    assert patient.tag == '2'
    assert patient.seed == 42
    assert patient.root == '/'


def test_dump_config_file(tmpdir):
    path = tmpdir.mkdir("tmp")
    patient = Experiment('hi', tag='2', seed=42, root=str(path))

    assert patient.name == 'hi'
    assert patient.tag == '2'
    assert patient.seed == 42
    assert patient.root == str(path)

    patient.write_config()
    with open(patient.config_file(), 'r') as f:
        d = load(f.read())
        assert patient.name == d['name']
        assert patient.tag == d['tag']
        assert patient.seed == d['seed']
        assert patient.root == d['root']
