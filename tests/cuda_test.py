from memory_profiler import profile

from random_walk_package import matrix_generator_gaussian_pdf
from random_walk_package import plot_walk
from random_walk_package.bindings.cuda.brownian_gpu import brownian_walk_gpu


@profile
def test_brownian_walk_gpu():
    T = 400
    W = 2 * T + 1
    H = 2 * T + 1
    S = 15

    start_x = 8 * T // 5
    start_y = 8 * T // 5
    end_x = T // 4
    end_y = T // 3

    kernel = matrix_generator_gaussian_pdf(2 * S + 1, 2 * S + 1, sigma=10.0)

    result_ptr = brownian_walk_gpu(kernel, S=S, T=T, W=W, H=H, start_x=start_x, start_y=start_y, end_x=end_x,
                                   end_y=end_y)

    plot_walk(result_ptr, W, H)


test_brownian_walk_gpu()
