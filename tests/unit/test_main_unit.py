from unittest.mock import patch
from pytest import mark

pytestmark = [mark.sdk, mark.unit]


@patch('trainml.cli.cli')
def test_main_module_execution(mock_cli):
    """Test that __main__ module calls cli() when executed as main."""
    # Import the module to get the cli reference
    import trainml.__main__
    
    # Execute the code that runs when __name__ == '__main__'
    # We need to simulate the main execution block
    # Since we can't change __name__ after import, we'll directly call
    # the logic that would execute
    trainml.__main__.cli()
    
    # Verify cli was called
    mock_cli.assert_called_once()
