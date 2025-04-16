# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands
- Install: `poetry install --with dev`
- Run all tests: `make test` or `pytest && pip3 install . && python3 tests/integration.py`
- Run single test: `pytest tests/test_file.py::test_function_name -v`
- Format code: `make format` or `black . && ruff check . --fix`
- Lint code: `make lint` or `ruff check .`

## Style Guidelines
- Line length: 99 characters (configured for both Black and Ruff)
- Imports: stdlib first, third-party next, local imports last
- Naming: CamelCase for classes, snake_case for functions/vars, UPPER_CASE for constants
- Documentation: NumPy-style docstrings with examples, parameters, returns
- Type hints: Use throughout codebase
- Error handling: Validate inputs, use ValueError with descriptive messages
- Use operator overloading (`__add__`, `__mul__`, etc.) and custom operators (`@` for sampling)
- Tests: Descriptive names, unit tests match module structure, use hypothesis for property testing