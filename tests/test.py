# debugging: gdb --args python -m tests.test
from tests.biased_test import biased_test

if __name__ == "__main__":
    # brownian_test.demonstrate_all_functionality()
    biased_test()
    # plot_walk_from_json("/home/omar/CLionProjects/random-walks/resources/biased.json", 'Biased Walk')
    # test_time_walker()
