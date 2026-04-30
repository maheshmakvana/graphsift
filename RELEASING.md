# Releasing graphsift

## One-time setup

1. Create the `graphsift` project on PyPI if it does not already exist.
2. In PyPI, configure a trusted publisher for this GitHub repository and the workflow file `.github/workflows/publish.yml`.
3. In GitHub, create an environment named `pypi` if you want environment protection before publish.

## Release flow

1. Bump `graphsift/_version.py`.
2. Commit and push the change.
3. Create a GitHub release for the new version.
4. The `Publish to PyPI` workflow will build the sdist and wheel, run `twine check`, and publish to PyPI.

## Manual local verification

If you want to verify locally before cutting a release:

```bash
python -m pip install --upgrade build twine
rm -rf build dist *.egg-info
pyproject-build
twine check dist/*
```

Use `pyproject-build` instead of `python -m build` because a top-level `build/` directory can shadow the module entry point.
