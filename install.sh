pip install --upgrade pip
pip install -r requirements.txt

# install overcooked_ai
cd overcooked_ai
pip install -e .

python setup.py develop
