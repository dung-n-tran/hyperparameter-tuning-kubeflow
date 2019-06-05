import os
import requests
import json
import subprocess
import yaml
import time


RANDOM_SPEC = """suggestionAlgorithm: "random"
    requestNumber: {}
"""

BAYESIAN_SPEC = """suggestionAlgorithm: "bayesianoptimization"
    suggestionParameters:
      -
          name: "burn_in"
          value: "{1}"
    requestNumber: {0}
"""

# GRID_SPEC ="""suggestionAlgorithm: "grid"
#     suggestionParameters:
#       -
#           name: "DefaultGrid"
#           value: "3"
# """


def generate_hyperparameter_tuning_yaml(study_name, search_type, request_number):
    if not study_name:
        raise ValueError("Study name cannot be None")

    template_file = os.path.join('kubeflow', 'template.train.yaml')

    if search_type == 'random':
        studyjob_name = '{}-random'.format(study_name)
        studyjob_file = '{}.yaml'.format(studyjob_name)

        generate_yaml_from_template(
            template_file,
            studyjob_file,
            **{
                'NAME': studyjob_name,
                'SPEC': RANDOM_SPEC.format(request_number)
            }
        )
    elif search_type == 'bayesian':
        studyjob_name = '{}-bayesian'.format(study_name)
        studyjob_file = '{}.yaml'.format(studyjob_name)

        generate_yaml_from_template(
            template_file,
            studyjob_file,
            **{
                'NAME': studyjob_name,
                'SPEC': BAYESIAN_SPEC.format(request_number, request_number // 2)  # second param is 'burn-in'
            }
        )
    else:
        raise ValueError("Currently, this util only supports 'random' and 'bayesian' search type")

    print("Studyjob spec has generated. To start, run 'kubectl create -f {}'".format(studyjob_file))
    return studyjob_name, studyjob_file


def generate_model_testing_yaml(study_name, study_id, model_id):
    tfjob_name = "{}-test".format(study_name)
    tfjob_file = "{}.yaml".format(tfjob_name)

    generate_yaml_from_template(
        os.path.join("kubeflow", "template.test.yaml"),
        tfjob_file,
        **{
            'NAME': tfjob_name,
            'MODEL_DIR': "\"/tmp/tensorflow/{0}/{1}_model\"".format(study_id, model_id)
        }
    )
    return tfjob_name, tfjob_file


def generate_yaml_from_template(template, filename, **kwargs):
    with open(template, 'r') as rf:
        tmp = rf.read()
        if kwargs is not None:
            for k, v in kwargs.items():
                tmp = tmp.replace('{{{}}}'.format(k), v)
        with open(filename, 'w') as wf:
            wf.write(tmp)


def connect():
    """Check port-forwarding and re-tunneling if connection lost"""
    return subprocess.Popen("kubectl port-forward svc/vizier-core-rest 6790:80", shell=True)


def get_study_metrics(study_id, worker_ids, metrics_names=None):
    """Get metrics

    Args:
        study_id (str):
        worker_ids (list of str):
        metrics_names (list of str):

    Returns:
        (json): metrics
    """
    data = {
        "study_id": study_id,
        "worker_ids": worker_ids,
        "metrics_names": metrics_names
    }
    json_data = json.dumps(data)

    return _post("http://localhost:6790/api/Manager/GetMetrics", json_data)


def _post(url, json_data):
    resp = requests.post(
        url, data=json_data, headers={'Content-type': 'application/json'}
    )
    resp.raise_for_status()

    return resp.json()


def get_study_result(studyjob_name, verbose=True, result_dir=None):
    # Note, Parameter configs keys are duplicated in the result. Should not use when parse the result.
    study_result = subprocess.run(
        ['kubectl', 'describe', 'studyjob', studyjob_name],
        stdout=subprocess.PIPE
    ).stdout.decode('utf-8')
    if verbose:
        print(study_result, "\n\n")

    try:
        r = yaml.safe_load(study_result)
        print("Study name:", r['Name'])
        print("Study id:", r['Status']['Studyid'])
        print("Duration:", (r['Status']['Completion Time'] - r['Status']['Start Time']).total_seconds())
        print("Best trial:", r['Status']['Best Trial Id'])
        print("Best worker:", r['Status']['Best Worker Id'])
        print("Best validation acc:", r['Status']['Best Objective Value'])
    except KeyError:
        print("Study is still running or not completed")
        return

    # Cache (backup) results as yaml file for later use
    if result_dir:
        os.makedirs(result_dir, exist_ok=True)

        result_filename = os.path.join(result_dir, "{}-result.yaml".format(studyjob_name))
        with open(result_filename, 'w') as f:
            f.write(study_result)

        print("Result is saved to {}".format(result_filename))

    return r


def get_best_model_id(study_result, timeout_sec=10.0):
    while timeout_sec > 0.0:
        try:
            model_id = get_study_metrics(
                study_id=study_result['Status']['Studyid'],
                worker_ids=[study_result['Status']['Best Worker Id']],
                metrics_names=["model_id"]
            )['metrics_log_sets'][0]['metrics_logs'][0]['values'][0]['value']
            return model_id
        except requests.ConnectionError:
            connect()
            timeout_sec -= 0.5
            time.sleep(0.5)

    raise TimeoutError("Connection timeout")
