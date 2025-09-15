~~~bash
python -m venv myenv
~~~

~~~bash
source myenv/bin/activate
~~~

~~~bash
pip install .
~~~

~~~bash
cp build/lib/librandom_walk.so random_walk_package/
~~~

~~~bash
python -m tests.test
~~~