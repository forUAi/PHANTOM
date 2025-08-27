# Contributing to PHANTOM

We love your input! We want to make contributing to PHANTOM as easy and transparent as possible.

## Development Process

1. Fork the repo and create your branch from `main`
2. Install development dependencies: `pip install -e ".[dev]"`
3. Make your changes
4. Add tests for any new functionality
5. Ensure tests pass: `pytest tests/`
6. Run linters: `make lint`
7. Submit a Pull Request!

## Code Style

- We use Black for formatting (100 char line length)
- We use Ruff for linting
- We use MyPy for type checking
- All new code should have type hints

## Testing

- Write tests for new features
- Maintain >80% code coverage
- Use pytest markers for hardware-specific tests

## Adding a New Backend

1. Create `phantom_core/exec/targets/your_backend.py`
2. Inherit from `TargetRunner` base class
3. Implement `run_ops` method
4. Add tests in `tests/test_your_backend.py`
5. Update documentation

## Pull Request Process

1. Update README.md with details of changes if needed
2. Update the version number in `pyproject.toml`
3. The PR will be merged once you have approval from maintainers

## License

By contributing, you agree that your contributions will be licensed under MIT License.