from pytest import mark

import trainml.exceptions as specimen

pytestmark = [mark.sdk, mark.unit]


def test_api_error():
    error = specimen.ApiError(400, dict(errorMessage="test message"))
    assert repr(error) == "ApiError(400, 'test message')"
    assert str(error) == "ApiError(400, 'test message')"


def test_job_error():
    error = specimen.JobError("failed", dict(id="id-1", status="failed"))
    assert (
        repr(error) == "JobError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert str(error) == "JobError(failed, {'id': 'id-1', 'status': 'failed'})"


def test_dataset_error():
    error = specimen.DatasetError("failed", dict(id="id-1", status="failed"))
    assert (
        repr(error)
        == "DatasetError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert (
        str(error)
        == "DatasetError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
