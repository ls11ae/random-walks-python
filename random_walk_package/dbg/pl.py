import matplotlib.pyplot as plt

# New set of points
new_points = [
    (1, 4),
    (1, 4),
    (2, 5),
    (1, 7),
    (1, 9),
    (1, 10),
    (2, 12),
    (2, 13),
    (2, 14),
    (2, 16),
    (3, 18),
    (4, 19),
    (4, 19),
    (6, 20),
    (7, 21),
    (9, 21),
    (10, 21),
    (11, 21),
    (13, 21),
    (14, 21),
    (16, 21),
    (17, 21),
    (17, 21),
    (19, 21),
    (21, 22),
    (22, 23),
    (23, 24),
    (24, 24),
    (25, 24),
    (25, 24),
]

# Extract x and y coordinates
x_new, y_new = zip(*new_points)

# Plot
plt.figure(figsize=(8, 8))
plt.plot(x_new, y_new, marker='o', linestyle='-')
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Trajectory Plot - Second Dataset")
plt.grid(True)
plt.show()
