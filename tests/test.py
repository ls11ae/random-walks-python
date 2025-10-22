# debugging: gdb --args python -m tests.test
from random_walk_package.bindings.plotter import plot_walk_from_json

if __name__ == "__main__":
    # brownian_test.demonstrate_all_functionality()
    plot_walk_from_json("/home/omar/CLionProjects/random-walks/resources/biased.json", 'Biased Walk')
    # test_time_walker()
