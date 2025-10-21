# debugging: gdb --args python -m tests.test
from tests.mixed_walk_test import test_time_walker

if __name__ == "__main__":
    # brownian_test.demonstrate_all_functionality()
    test_time_walker()
