# debugging: gdb --args python -m tests.test
from tests import correlated_test

if __name__ == "__main__":
    # brownian_test.demonstrate_all_functionality()
    correlated_test.test_correlated_multistep()
