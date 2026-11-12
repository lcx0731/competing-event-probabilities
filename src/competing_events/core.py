"""Aalen-Johansen incidence for baseline cohorts with right censoring.

Event-before-censor handling is explicit at tied integer observation times.
Fractions make computations exact for the supplied counts, not statistically
certain. No delayed entry, covariates, or confidence intervals are implemented.
"""

from bisect import bisect_right
from dataclasses import dataclass
from fractions import Fraction
import re


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f'{name} must be an integer from {low} to {high}')
    return value


def _causes(causes):
    if not isinstance(causes, (list, tuple)) or not 1 <= len(causes) <= 8:
        raise ValueError('provide 1 to 8 cause labels')
    if any(not isinstance(s, str) or re.fullmatch('[A-Za-z][A-Za-z0-9_-]{0,31}', s) is None for s in causes):
        raise ValueError('cause labels must be short ASCII identifiers')
    if len(set(causes)) != len(causes):
        raise ValueError('cause labels must be unique')
    return tuple(causes)


@dataclass(frozen=True)
class EventRow:
    time: int
    events: tuple
    censored: int

    def __post_init__(self):
        _integer(self.time, 0, 10**9, 'time')
        _integer(self.censored, 0, 10**9, 'censored')
        if not isinstance(self.events, (list, tuple)) or not 1 <= len(self.events) <= 8:
            raise ValueError('events must contain 1 to 8 integer counts')
        events = tuple(_integer(x, 0, 10**9, 'event count') for x in self.events)
        if sum(events)+self.censored == 0:
            raise ValueError('each row must contain an event or censoring observation')
        object.__setattr__(self, 'events', events)


@dataclass(frozen=True)
class EventTable:
    initial: int
    causes: tuple
    rows: tuple

    def __post_init__(self):
        _integer(self.initial, 1, 10**9, 'initial cohort size')
        causes = _causes(self.causes)
        if not isinstance(self.rows, (list, tuple)) or not 1 <= len(self.rows) <= 200:
            raise ValueError('provide 1 to 200 observation-time rows')
        at_risk, previous = self.initial, -1
        for row in self.rows:
            if not isinstance(row, EventRow) or len(row.events) != len(causes):
                raise ValueError('rows must have one count per declared cause')
            if row.time <= previous:
                raise ValueError('row times must be strictly increasing')
            removed = sum(row.events)+row.censored
            if removed > at_risk:
                raise ValueError('events plus censoring exceed the risk set')
            at_risk -= removed
            previous = row.time
        if at_risk != 0:
            raise ValueError('all initial members need an event or censoring record; include administrative censoring')
        object.__setattr__(self, 'causes', causes)
        object.__setattr__(self, 'rows', tuple(self.rows))


@dataclass(frozen=True)
class Point:
    time: int
    at_risk: int
    events: tuple
    censored: int
    remaining: int
    survival: Fraction
    incidence: tuple
    net_risk: tuple
    cause_hazards: tuple

    def to_dict(self):
        return {'time': self.time, 'at_risk': self.at_risk, 'events': list(self.events),
                'censored': self.censored, 'remaining': self.remaining,
                'survival': str(self.survival), 'cumulative_incidence': list(map(str, self.incidence)),
                'net_risk': list(map(str, self.net_risk)), 'cause_hazards': list(map(str, self.cause_hazards))}


@dataclass(frozen=True)
class Estimate:
    table: EventTable
    points: tuple

    def at(self, time):
        """Right-continuous estimate within observed support; no extrapolation."""
        _integer(time, 0, self.points[-1].time, 'query time')
        index = bisect_right([p.time for p in self.points], time)-1
        if index < 0:
            return {'time': time, 'survival': '1', 'cumulative_incidence': ['0']*len(self.table.causes), 'net_risk': ['0']*len(self.table.causes)}
        p = self.points[index]
        return {'time': time, 'survival': str(p.survival), 'cumulative_incidence': list(map(str, p.incidence)), 'net_risk': list(map(str, p.net_risk))}

    def to_dict(self):
        return {'initial': self.table.initial, 'causes': list(self.table.causes),
                'support_end': self.points[-1].time,
                'tie_convention': 'Events use the pre-removal risk set; tied censoring is removed afterward.',
                'arithmetic': 'Exact rational point estimates from supplied integer counts; not confidence bounds.',
                'points': [p.to_dict() for p in self.points],
                'scope': 'All subjects enter at baseline. Interpreting incidence as population probability requires suitable independent censoring and follow-up support.'}


def estimate(table):
    if not isinstance(table, EventTable):
        raise ValueError('table must be an EventTable')
    k = len(table.causes)
    survival, incidence = Fraction(1), [Fraction(0)]*k
    net_survival = [Fraction(1)]*k
    at_risk, points = table.initial, []
    for row in table.rows:
        hazards = tuple(Fraction(d, at_risk) for d in row.events)
        before = survival
        incidence = [f+before*h for f, h in zip(incidence, hazards)]
        survival = before*(1-sum(hazards))
        net_survival = [s*(1-h) for s, h in zip(net_survival, hazards)]
        remaining = at_risk-sum(row.events)-row.censored
        points.append(Point(row.time, at_risk, row.events, row.censored, remaining,
                            survival, tuple(incidence), tuple(1-s for s in net_survival), hazards))
        at_risk = remaining
    return Estimate(table, tuple(points))


def from_records(records, causes):
    """Aggregate 1 to 10,000 {'time': integer, 'cause': label or None} rows.

    None denotes right censoring. Every record belongs to the baseline cohort.
    """
    causes = _causes(causes)
    if not isinstance(records, (list, tuple)) or not 1 <= len(records) <= 10000:
        raise ValueError('provide 1 to 10000 individual records')
    grouped = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {'time', 'cause'}:
            raise ValueError('records require exactly time and cause')
        time = _integer(record['time'], 0, 10**9, 'time')
        cause = record['cause']
        if cause is not None and cause not in causes:
            raise ValueError('unknown event cause')
        events, censored = grouped.setdefault(time, [[0]*len(causes), 0])
        if cause is None:
            grouped[time][1] += 1
        else:
            events[causes.index(cause)] += 1
    rows = [EventRow(time, values[0], values[1]) for time, values in sorted(grouped.items())]
    return EventTable(len(records), causes, rows)
