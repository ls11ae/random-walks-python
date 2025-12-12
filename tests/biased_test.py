from random_walk_package.core.BiasedWalker import BiasedWalker


def test_biased():
    T = 150
    size = 200
    bias_offsets1 = [(9, 0)] * (T // 3)
    bias_offsets2 = [(0, 0)] * (T // 3)
    bias_offsets3 = [(-9, 0)] * (T - len(bias_offsets1) - len(bias_offsets2))
    biases = bias_offsets1 + bias_offsets2 + bias_offsets3
    print(len(biases))
    walker = BiasedWalker(S=7, W=size, H=size, T=T)
    walker.generate(bias_offsets=biases, start_x=100, start_y=23)
    walk = walker.backtrace(end_x=100, end_y=180)

    assert len(walk) == T
    assert tuple(walk[0]) == (100, 23)
    assert tuple(walk[-1]) == (100, 180)
