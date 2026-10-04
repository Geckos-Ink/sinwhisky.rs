#!/usr/bin/env python3
"""Numerical checks for docs/PAPER_EQ_COMPRESSION.md.

Usage: eq_compression_checks.py

Standard library only, deterministic, no audio or measured data. It prints Table 1
of the paper (the fraction of a tone's windowed energy on the far side of a band
edge, from the window kernels of PAPER_FAFT.md (42)) and asserts the identities the
paper states, each against an independent computation:

- the ascending pass of a Cholesky solve gives the explained energy, and a banded
  Gram matrix factors without fill-in (Proposition 3);
- the state of a one-pole analog filter carried across rotations by the closed
  form equals a Runge-Kutta integration of the same curve (Proposition 5);
- the transform of a spiral rotation and its endpoint form equal Gauss-Legendre
  quadrature (Proposition 6);
- the gradient-tone normal equations built from kernel and transform derivatives
  equal a direct numerical least-squares fit, images included, and recover the
  frequency gradient of a chirp (Proposition 8);
- the periodic Hann window sums to one at a hop of half its length.

It also reports how much truncating the Gram matrix of a 55 Hz harmonic series to
the reach of the Hann kernel changes the jointly fitted amplitudes. Exit status is
nonzero if a check fails.
"""
import cmath
import math
import random


def gauss_legendre(n):
    """Nodes and weights of the n-point Gauss-Legendre rule on [-1, 1]."""
    nodes, weights = [], []
    for i in range(1, n + 1):
        x = math.cos(math.pi * (i - 0.25) / (n + 0.5))
        for _ in range(100):
            p0, p1 = 1.0, x
            for k in range(2, n + 1):
                p0, p1 = p1, ((2 * k - 1) * x * p1 - (k - 1) * p0) / k
            dp = n * (x * p1 - p0) / (x * x - 1)
            step = p1 / dp
            x -= step
            if abs(step) < 1e-16:
                break
        nodes.append(x)
        weights.append(2 / ((1 - x * x) * dp * dp))
    return list(zip(nodes, weights))


RULE = gauss_legendre(24)


def integrate(f, a, b, panels):
    """Composite Gauss-Legendre integral of f on [a, b]."""
    h = (b - a) / panels
    total = 0
    for k in range(panels):
        mid = a + (k + 0.5) * h
        total += sum(w * f(mid + x * h / 2) for x, w in RULE) * h / 2
    return total


def solve(matrix, rhs):
    """Dense Gaussian elimination with partial pivoting (small systems only)."""
    n = len(rhs)
    m = [row[:] + [rhs[i]] for i, row in enumerate(matrix)]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        for r in range(n):
            if r != i:
                f = m[r][i] / m[i][i]
                for c in range(i, n + 1):
                    m[r][c] -= f * m[i][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def cholesky(g):
    n = len(g)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = g[i][j] - sum(lower[i][k] * lower[j][k] for k in range(j))
            lower[i][j] = math.sqrt(s) if i == j else s / lower[j][j]
    return lower


def two_way(lower, b):
    """Ascending (forward) then descending (back) substitution."""
    n = len(b)
    z = []
    for i in range(n):
        z.append((b[i] - sum(lower[i][k] * z[k] for k in range(i))) / lower[i][i])
    x = [0.0] * n
    for i in reversed(range(n)):
        x[i] = (z[i] - sum(lower[k][i] * x[k] for k in range(i + 1, n))) / lower[i][i]
    return z, x


# ------------------------------------------------------------------ kernels


def sinc(x):
    return 1.0 if x == 0 else math.sin(math.pi * x) / (math.pi * x)


def k_rect(x):
    """Normalized rectangular kernel, x in bins of 2*pi/T."""
    return sinc(x)


def k_hann(x):
    """Normalized Hann kernel, PAPER_FAFT.md (42)."""
    return sinc(x) + 0.5 * sinc(x - 1) + 0.5 * sinc(x + 1)


def far_side(kernel, total, s, upper, tail):
    """Fraction of |k|^2 beyond s bins: Table 1 of the paper."""
    return (integrate(lambda x: kernel(x) ** 2, s, upper, int(2 * (upper - s))) + tail) / total


def table_1():
    edges = (0.5, 1, 2, 4, 8, 16)
    upper = 2000.0  # integer, so the rectangular tail is 1/(2 pi^2 U) to O(U^-3)
    rect = [far_side(k_rect, 1.0, s, upper, 1 / (2 * math.pi ** 2 * upper)) for s in edges]
    # Parseval: the normalized Hann kernel has integral of |k|^2 = 3/2; its tail
    # beyond 200 bins is below 1e-13 because |k| decays like x^-3.
    hann = [far_side(k_hann, 1.5, s, 200.0, 0.0) for s in edges]
    for s, r in zip(edges, rect):
        if s >= 2:
            assert abs(r * 2 * math.pi ** 2 * s - 1) < 0.02, 'rectangular tail is 1/(2 pi^2 s)'
    print('Table 1. Fraction of a tone\'s windowed energy beyond an edge s bins away')
    print('  s (bins)    ' + ''.join(f'{s:>18}' for s in edges))
    for name, values in (('rectangular', rect), ('Hann', hann)):
        cells = ''.join(f'{v:>9.3g} {10 * math.log10(v):6.2f} dB' for v in values)
        print(f'  {name:<12}{cells}')
    return rect, hann


# --------------------------------------------------------- Proposition 3


def check_two_way():
    random.seed(3)
    rows, cols = 40, 6
    atoms = [[random.gauss(0, 1) for _ in range(cols)] for _ in range(rows)]
    y = [random.gauss(0, 1) for _ in range(rows)]
    g = [[sum(a[i] * a[j] for a in atoms) for j in range(cols)] for i in range(cols)]
    b = [sum(a[i] * v for a, v in zip(atoms, y)) for i in range(cols)]
    z, x = two_way(cholesky(g), b)
    residual = sum((v - sum(a[i] * x[i] for i in range(cols))) ** 2 for a, v in zip(atoms, y))
    energy = sum(v * v for v in y)
    assert abs(energy - sum(v * v for v in z) - residual) < 1e-10 * energy
    # A banded matrix factors without fill-in: entries outside the band stay 0.
    n, reach = 12, 2
    banded = [[(4.0 if i == j else 1.0 / (1 + abs(i - j))) if abs(i - j) <= reach else 0.0
               for j in range(n)] for i in range(n)]
    lower = cholesky(banded)
    assert all(lower[i][j] == 0.0 for i in range(n) for j in range(i) if i - j > reach)


def harmonic_chain():
    """Amplitude change from truncating the Hann Gram beyond a reach in bins."""
    fs, n, f0, count = 44100.0, 1024, 55.0, 20
    bins = [k * f0 * n / fs for k in range(1, count + 1)]
    gram = [[(k_hann(p - q) + k_hann(p + q)) / 2 for q in bins] for p in bins]  # (F45)
    random.seed(7)
    alpha = [math.cos(random.uniform(0, 2 * math.pi)) / k for k in range(1, count + 1)]
    b = [sum(gram[i][j] * alpha[j] for j in range(count)) for i in range(count)]
    _, exact = two_way(cholesky(gram), b)
    assert max(abs(e - a) for e, a in zip(exact, alpha)) < 1e-10
    changes = []
    for reach in (4, 8, 16):
        truncated = [[gram[i][j] if abs(bins[i] - bins[j]) <= reach else 0.0 for j in range(count)]
                     for i in range(count)]
        _, approx = two_way(cholesky(truncated), b)
        changes.append(math.sqrt(sum((a - e) ** 2 for a, e in zip(approx, exact))
                                 / sum(e * e for e in exact)))
    assert changes[0] > changes[1] > changes[2]
    print(f'55 Hz series, {count} harmonics {bins[0]:.2f} bins apart, Hann Gram truncated beyond'
          f' 4 / 8 / 16 bins: relative RMS amplitude change '
          + ' / '.join(f'{c:.2e}' for c in changes))


# ------------------------------------------------- Propositions 5 and 6


def e0(delta, length):
    if abs(delta * length) < 1e-8:
        return length
    return (cmath.exp(1j * delta * length) - 1) / (1j * delta)


def e1(delta, length):
    if abs(delta * length) < 1e-6:
        return length * length / 2
    return (length * cmath.exp(1j * delta * length) - e0(delta, length)) / (1j * delta)


def rotation_transform(omega, length, c, r, phi, beta, d):
    """PAPER_FAFT.md (21), valid at complex omega."""
    z0 = r * cmath.exp(1j * phi)
    return (c * e0(-omega, length)
            + (z0 * e0(beta - omega, length) - z0.conjugate() * e0(-beta - omega, length)) / 2j
            + d * e1(-omega, length))


def check_filter():
    pole = complex(-0.07, 0.31)
    rotations = [(5.0, 0.3, 1.2, 0.4, -0.7, 0.0), (3.0, -0.5, 0.0, 0.0, 0.0, 0.2),
                 (7.0, 1.1, 0.8, -1.3, 0.45, 0.0)]
    closed = numeric = complex(0.3, -0.1)
    for length, c, r, phi, beta, d in rotations:
        closed = cmath.exp(pole * length) * (closed + rotation_transform(-1j * pole, length, c, r, phi, beta, d))
        steps = 4000
        h = length / steps

        def slope(u, state):
            return pole * state + c + r * math.sin(phi + beta * u) + d * u

        for k in range(steps):
            u = k * h
            k1 = slope(u, numeric)
            k2 = slope(u + h / 2, numeric + h / 2 * k1)
            k3 = slope(u + h / 2, numeric + h / 2 * k2)
            k4 = slope(u + h, numeric + h * k3)
            numeric += h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    assert abs(closed - numeric) < 1e-10 * abs(closed)


def check_spiral():
    length, c, r, phi, sigma, beta = 6.0, 0.4, 1.3, 0.7, -0.15, 0.9
    lam = complex(sigma, beta)
    z0 = r * cmath.exp(1j * phi)

    def value(u):
        return c + (z0 * cmath.exp(lam * u)).imag

    def slope(u):
        return (z0 * lam * cmath.exp(lam * u)).imag

    for omega in (0.0, 0.5, 0.9, 1.7, -2.3):
        def moment(mu):
            z = mu - 1j * omega
            return length if abs(z) < 1e-14 else (cmath.exp(z * length) - 1) / z

        quadrature = integrate(lambda u: value(u) * cmath.exp(-1j * omega * u), 0, length, 64)
        closed = c * moment(0) + (z0 * moment(lam) - z0.conjugate() * moment(lam.conjugate())) / 2j
        q = cmath.exp(-1j * omega * length)
        jump = (value(length) - c) * q - (value(0) - c)
        slope_jump = slope(length) * q - slope(0)
        endpoint = c * moment(0) - (slope_jump + (1j * omega - 2 * sigma) * jump) / (
            (1j * omega - lam) * (1j * omega - lam.conjugate()))
        assert abs(closed - quadrature) < 1e-12 and abs(endpoint - quadrature) < 1e-12


# ----------------------------------------------------------- Proposition 8


def check_gradients():
    t = 256.0
    weight = t / 2

    def hann(u):
        return 0.5 + 0.5 * math.cos(2 * math.pi * u / t)

    def window_integral(f):
        return integrate(f, -t / 2, t / 2, 96)

    def kernel_derivative(j, nu):
        return window_integral(lambda u: hann(u) * (-1j * u) ** j * cmath.exp(-1j * nu * u)) / weight

    def fit(signal, omega):
        """Gradient-tone coefficients c_0, c_1, c_2 from (18)-(20) of the paper."""
        same = [kernel_derivative(j, 0.0) for j in range(5)]
        image = [kernel_derivative(j, 2 * omega) for j in range(5)]
        rows, rhs = [], []
        for m in range(3):
            data = (1j) ** m * window_integral(
                lambda u: signal(u) * hann(u) * (-1j * u) ** m * cmath.exp(-1j * omega * u)) / weight
            re_row, im_row = [], []
            for n in range(3):
                g, h = (1j) ** (n + m) * same[n + m], (1j) ** (n + m) * image[n + m]
                re_row += [(g + h).real, (1j * (g - h)).real]
                im_row += [(g + h).imag, (1j * (g - h)).imag]
            rows += [re_row, im_row]
            rhs += [data.real, data.imag]
        x = solve(rows, rhs)
        return [complex(x[2 * n], x[2 * n + 1]) for n in range(3)]

    # Normal equations against a direct least-squares fit, 2.3 bins from DC.
    omega = 2.3 * 2 * math.pi / t

    def mixture(u):
        return ((1 + 0.003 * u) * math.cos(0.9 * omega * u + 2e-5 * u * u + 0.4)
                + 0.2 * math.sin(1.7 * omega * u))

    closed = fit(mixture, omega)
    atoms = ([lambda u, m=m: u ** m * math.cos(omega * u) for m in range(3)]
             + [lambda u, m=m: u ** m * math.sin(omega * u) for m in range(3)])
    gram = [[window_integral(lambda u: atoms[i](u) * atoms[j](u) * hann(u)) for j in range(6)]
            for i in range(6)]
    data = [window_integral(lambda u: atoms[i](u) * mixture(u) * hann(u)) for i in range(6)]
    x = solve(gram, data)
    direct = [complex(x[m] / 2, -x[3 + m] / 2) for m in range(3)]
    assert max(abs(a - b) / abs(b) for a, b in zip(closed, direct)) < 1e-10

    # Frequency gradient of a constant-amplitude chirp analysed at its centre.
    omega, gamma = 0.31, 8e-6  # gamma*T^2/8 = 0.066 rad
    c0, _, c2 = fit(lambda u: math.cos(omega * u + 0.5 * gamma * u * u + 0.6), omega)
    estimate = 2 * (c2 / c0).imag
    assert abs(estimate - gamma) < 1e-3 * gamma, (estimate, gamma)
    print(f'Chirp on a {t:.0f}-sample Hann window, gamma*T^2/8 = {gamma * t * t / 8:.3f} rad:'
          f' relative error of 2*Im(c2/c0) {abs(estimate - gamma) / gamma:.1e}')


def main():
    table_1()
    check_two_way()
    harmonic_chain()
    check_filter()
    check_spiral()
    check_gradients()
    theta = 0.37
    assert abs((0.5 - 0.5 * math.cos(theta)) + (0.5 - 0.5 * math.cos(theta + math.pi)) - 1) < 1e-15
    print('All checks passed.')


if __name__ == '__main__':
    main()
