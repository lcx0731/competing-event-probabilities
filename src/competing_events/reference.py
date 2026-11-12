"""Independent individual-record transition-matrix product, for small checks."""

from fractions import Fraction


def matrix_reference(records, causes):
    """Assume validated records; recompute each risk set directly, without tables."""
    if not 1 <= len(records) <= 100:
        raise ValueError('reference supports 1 to 100 records')
    size = len(causes)+1
    state = [Fraction(1)]+[Fraction(0)]*(size-1)
    output = []
    for time in sorted({r['time'] for r in records}):
        at_risk = sum(r['time'] >= time for r in records)
        matrix = [[Fraction(i == j) for j in range(size)] for i in range(size)]
        for j, cause in enumerate(causes, 1):
            count = sum(r['time'] == time and r['cause'] == cause for r in records)
            matrix[0][j] = Fraction(count, at_risk)
        matrix[0][0] = 1-sum(matrix[0][1:])
        state = [sum(state[i]*matrix[i][j] for i in range(size)) for j in range(size)]
        output.append((time, tuple(state)))
    return output
