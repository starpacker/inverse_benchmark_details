# Test Files

This directory contains unit tests and test data for validating the reconstruction code.

## Structure

- `test_*.py`: Unit test files for individual functions
- `test_data/`: Standard test data in pickle format
- `agents/`: Function decomposition files (if available)
- `verification_utils.py`: Utilities for test verification

## Running Tests

```bash
# Run all tests
python -m pytest test/

# Run specific test
python test/test_function_name.py
```

## Test Data

Test data files (`*.pkl`) contain input/output pairs for validating function behavior.
