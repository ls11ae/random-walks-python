python -m venv myenv 
source myenv/bin/activate
pip install .
cp build/lib/librandomwalk.so random_walk_package/
python -m tests.test
