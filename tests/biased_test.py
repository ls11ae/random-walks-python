from random_walk_package.core.BiasedWalker import BiasedWalker


def test_biased():
    T = 150
    size = 400
    bias_offsets1 = [(9, 0)] * (T // 3)
    bias_offsets2 = [(0, 0)] * (T // 3)
    bias_offsets3 = [(-9, 0)] * (T - len(bias_offsets1) - len(bias_offsets2))
    biases = bias_offsets1 + bias_offsets2 + bias_offsets3
    print(len(biases))
    walker = BiasedWalker(S=7, W=size, H=size, T=T)
    walker.generate(bias_offsets=biases, start_x=200, start_y=50)
    walk = walker.backtrace(end_x=200, end_y=280)

    assert len(walk) == T
    assert tuple(walk[0]) == (200, 50)
    assert tuple(walk[-1]) == (200, 280)
