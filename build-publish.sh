source .venv/bin/activate
rm -r dist/ build/ *.egg-info/
python3 setup.py sdist bdist_wheel
twine upload dist/* --verbose