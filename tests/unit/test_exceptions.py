from pytest import mark

import trainml.exceptions as specimen

pytestmark = [mark.sdk, mark.unit]


def test_trainml_exception():
    """Test TrainMLException base class."""
    error = specimen.TrainMLException("test message")
    assert error.message == "test message"
    assert repr(error) == "TrainMLException( 'test message')"
    assert str(error) == "TrainMLException('test message')"
    
    # Test with multiple args
    error2 = specimen.TrainMLException("test", "arg1", "arg2")
    assert error2.message == "test"


def test_api_error():
    """Test ApiError exception."""
    error = specimen.ApiError(400, dict(errorMessage="test message"))
    assert error.status == 400
    assert error.message == "test message"
    assert repr(error) == "ApiError(400, 'test message')"
    assert str(error) == "ApiError(400, 'test message')"
    
    # Test with message key instead of errorMessage
    error2 = specimen.ApiError(404, dict(message="not found"))
    assert error2.message == "not found"
    
    # Test with multiple args
    error3 = specimen.ApiError(500, dict(errorMessage="server error"), "extra")
    assert error3.message == "server error"


def test_job_error():
    """Test JobError exception."""
    error = specimen.JobError("failed", dict(id="id-1", status="failed"))
    assert error.status == "failed"
    assert error.message == dict(id="id-1", status="failed")
    assert (
        repr(error) == "JobError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert str(error) == "JobError(failed, {'id': 'id-1', 'status': 'failed'})"
    
    # Test with string data
    error2 = specimen.JobError("errored", "error string")
    assert error2.message == "error string"


def test_dataset_error():
    """Test DatasetError exception."""
    error = specimen.DatasetError("failed", dict(id="id-1", status="failed"))
    assert error.status == "failed"
    assert error.message == dict(id="id-1", status="failed")
    assert (
        repr(error)
        == "DatasetError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert (
        str(error)
        == "DatasetError(failed, {'id': 'id-1', 'status': 'failed'})"
    )


def test_model_error():
    """Test ModelError exception."""
    error = specimen.ModelError("failed", dict(id="id-1", status="failed"))
    assert error.status == "failed"
    assert error.message == dict(id="id-1", status="failed")
    assert (
        repr(error) == "ModelError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert str(error) == "ModelError(failed, {'id': 'id-1', 'status': 'failed'})"


def test_checkpoint_error():
    """Test CheckpointError exception."""
    error = specimen.CheckpointError("failed", dict(id="id-1", status="failed"))
    assert error.status == "failed"
    assert error.message == dict(id="id-1", status="failed")
    assert (
        repr(error)
        == "CheckpointError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert (
        str(error)
        == "CheckpointError(failed, {'id': 'id-1', 'status': 'failed'})"
    )


def test_volume_error():
    """Test VolumeError exception."""
    error = specimen.VolumeError("failed", dict(id="id-1", status="failed"))
    assert error.status == "failed"
    assert error.message == dict(id="id-1", status="failed")
    assert (
        repr(error) == "VolumeError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert str(error) == "VolumeError(failed, {'id': 'id-1', 'status': 'failed'})"


def test_connection_error():
    """Test ConnectionError exception."""
    error = specimen.ConnectionError("connection failed")
    assert error.message == "connection failed"
    assert repr(error) == "ConnectionError(connection failed)"
    assert str(error) == "ConnectionError(connection failed)"
    
    # Test with multiple args
    error2 = specimen.ConnectionError("test", "arg1", "arg2")
    assert error2.message == "test"


def test_specification_error():
    """Test SpecificationError exception."""
    error = specimen.SpecificationError("attr", "invalid value")
    assert error.attribute == "attr"
    assert error.message == "invalid value"
    assert repr(error) == "SpecificationError(attr, invalid value)"
    assert str(error) == "SpecificationError(attr, invalid value)"
    
    # Test with multiple args
    error2 = specimen.SpecificationError("attr", "test", "arg1")
    assert error2.attribute == "attr"
    assert error2.message == "test"


def test_node_error():
    """Test NodeError exception."""
    error = specimen.NodeError("failed", dict(id="id-1", status="failed"))
    assert error.status == "failed"
    assert error.message == dict(id="id-1", status="failed")
    assert (
        repr(error) == "NodeError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert str(error) == "NodeError(failed, {'id': 'id-1', 'status': 'failed'})"


def test_provider_error():
    """Test ProviderError exception."""
    error = specimen.ProviderError("failed", dict(id="id-1", status="failed"))
    assert error.status == "failed"
    assert error.message == dict(id="id-1", status="failed")
    assert (
        repr(error)
        == "ProviderError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert (
        str(error)
        == "ProviderError(failed, {'id': 'id-1', 'status': 'failed'})"
    )


def test_region_error():
    """Test RegionError exception."""
    error = specimen.RegionError("failed", dict(id="id-1", status="failed"))
    assert error.status == "failed"
    assert error.message == dict(id="id-1", status="failed")
    assert (
        repr(error) == "RegionError(failed, {'id': 'id-1', 'status': 'failed'})"
    )
    assert str(error) == "RegionError(failed, {'id': 'id-1', 'status': 'failed'})"
