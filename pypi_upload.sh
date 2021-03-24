#!/usr/bin/env bash
python3 setup.py sdist bdist_wheel #upload -r pypi
twine upload dist/*
rm -rf ./dist
rm -rf ./timesmash.egg-info
