# Running the tests

```
pytest --cov-report term-missing --cov=trainml --dist=loadscope -nauto --env=<dev, staging, prod>
```

## Authentication

Either modify your `~/.trainml/credentials.json` file to include the correct credentials for the specified environment or prefix the command with the necessary environment variables.

```
TRAINML_USER=<env user> TRAINML_KEY=<env key> pytest --cov-report term-missing --cov=trainml --dist=loadscope -nauto --env=<dev, staging, prod>
```

## Targetting specific tests

Filterable markers can be found in the `pyproject.toml` file.

To run all job tests:

```
pytest --cov-report term-missing --cov=trainml --dist=loadscope -nauto --env=dev -m jobs
```

To run all tests that do not create resources:

```
pytest --cov-report term-missing --cov=trainml --dist=loadscope -nauto --env=dev -m "not create"
```
