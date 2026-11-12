from fractions import Fraction as F
import random

import pytest

from competing_events import EventRow, EventTable, estimate, from_records
from competing_events.reference import matrix_reference


def test_hand_computed_incidence_and_net_risk():
    table = EventTable(10, ['A','B'], [EventRow(1,[2,3],2), EventRow(2,[1,0],2)])
    a, b = estimate(table).points
    assert (a.survival,a.incidence) == (F(1,2),(F(1,5),F(3,10)))
    assert (b.survival,b.incidence) == (F(1,3),(F(11,30),F(3,10)))
    assert b.net_risk == (F(7,15),F(3,10))
    assert a.at_risk == 10 and b.at_risk == 3 and b.remaining == 0


@pytest.mark.parametrize('seed', range(10))
def test_against_independent_record_matrix_product(seed):
    rng = random.Random(seed)
    records = [{'time':rng.randrange(6),'cause':rng.choice([None,'A','B','C'])} for _ in range(30)]
    actual = estimate(from_records(records,['A','B','C']))
    reference = matrix_reference(records,['A','B','C'])
    previous = [F(0)]*3
    for point, (time, state) in zip(actual.points, reference):
        assert point.time == time
        assert (point.survival,*point.incidence) == state
        assert point.survival+sum(point.incidence) == 1
        assert all(f >= p for f,p in zip(point.incidence,previous))
        assert all(f <= net for f,net in zip(point.incidence,point.net_risk))
        previous = point.incidence


def test_single_cause_reduces_to_kaplan_meier():
    curve = estimate(EventTable(10,['A'],[EventRow(1,[2],3),EventRow(3,[1],4)]))
    for point in curve.points:
        assert point.incidence[0] == point.net_risk[0] == 1-point.survival


def test_cause_specific_net_risks_need_not_sum_to_one():
    curve = estimate(EventTable(10,['A','B'],[EventRow(1,[5,0],0),EventRow(2,[0,5],0)]))
    p = curve.points[-1]
    assert p.incidence == (F(1,2),F(1,2))
    assert p.net_risk == (F(1,2),F(1))
    assert sum(p.net_risk) == F(3,2)


def test_tied_censoring_stays_in_event_denominator():
    p = estimate(EventTable(10,['A','B'],[EventRow(1,[2,3],5)])).points[0]
    assert p.cause_hazards == (F(1,5),F(3,10))
    assert p.survival == F(1,2)
    assert p.remaining == 0  # no one observed further does not imply S=0


def test_all_censored_has_no_identified_tail():
    curve = estimate(EventTable(100,['A','B'],[EventRow(0,[0,0],20),EventRow(5,[0,0],80)]))
    assert curve.points[-1].survival == 1
    assert curve.points[-1].incidence == (0,0)
    with pytest.raises(ValueError):
        curve.at(6)


def test_right_continuity_before_between_and_at_events():
    curve = estimate(EventTable(10,['A'],[EventRow(2,[2],0),EventRow(5,[2],6)]))
    assert curve.at(0)['cumulative_incidence'] == ['0']
    assert curve.at(1)['cumulative_incidence'] == ['0']
    assert curve.at(2)['cumulative_incidence'] == ['1/5']
    assert curve.at(4)['cumulative_incidence'] == ['1/5']
    assert curve.at(5)['cumulative_incidence'] == ['2/5']
    for t in [-1,6,True,2.5]:
        with pytest.raises(ValueError):
            curve.at(t)


def test_integer_time_zero_event_is_right_continuous():
    curve = estimate(EventTable(2,['A'],[EventRow(0,[1],0),EventRow(1,[0],1)]))
    assert curve.at(0)['cumulative_incidence'] == ['1/2']


def test_count_replication_invariance_and_large_cohort():
    base = estimate(EventTable(10,['A','B'],[EventRow(1,[2,3],2),EventRow(2,[1,0],2)]))
    multiplier = 10**8
    large = estimate(EventTable(10*multiplier,['A','B'],[EventRow(1,[2*multiplier,3*multiplier],2*multiplier),EventRow(2,[multiplier,0],2*multiplier)]))
    assert [p.incidence for p in base.points] == [p.incidence for p in large.points]
    assert [p.survival for p in base.points] == [p.survival for p in large.points]


def test_unused_causes_are_retained():
    p = estimate(EventTable(10,['A','absent'],[EventRow(1,[2,0],8)])).points[0]
    assert p.incidence[1] == p.net_risk[1] == 0


def test_record_aggregation_and_order_invariance():
    records = [{'time':2,'cause':None},{'time':1,'cause':'B'},{'time':1,'cause':None},{'time':2,'cause':'A'}]
    expected = EventTable(4,['A','B'],[EventRow(1,[0,1],1),EventRow(2,[1,0],1)])
    assert from_records(records,['A','B']) == expected
    assert from_records(records[::-1],['A','B']) == expected


def test_balanced_early_censoring_preserves_constructed_curve():
    complete = estimate(EventTable(1000,['A','B'],[EventRow(1,[200,300],0),EventRow(2,[100,150],250)]))
    balanced = estimate(EventTable(1000,['A','B'],[EventRow(0,[0,0],300),EventRow(1,[140,210],0),EventRow(2,[70,105],175)]))
    assert complete.at(1) == balanced.at(1)
    assert complete.at(2) == balanced.at(2)


def test_two_latent_worlds_same_observed_records_different_incidence():
    world_a = [{'time':1,'cause':'A'}]*3+[{'time':1,'cause':'B'}]*2+[{'time':2,'cause':None}]*5
    world_b = [{'time':2,'cause':None}]*3+world_a[3:]
    observe = lambda records: [{'time':0,'cause':None} if i < 3 else row for i,row in enumerate(records)]
    assert observe(world_a) == observe(world_b)
    a = estimate(from_records(world_a,['A','B'])).at(2)
    b = estimate(from_records(world_b,['A','B'])).at(2)
    assert a['cumulative_incidence'][0] == '3/10'
    assert b['cumulative_incidence'][0] == '0'


@pytest.mark.parametrize('time,events,censored', [(-1,[1],0),(1.2,[1],0),(True,[1],0),(1,[-1],2),(1,[True],0),(1,[1.0],0),(1,[],1),(1,[0],0),(1,[1],-1),(1,[1],False)])
def test_invalid_event_row(time,events,censored):
    with pytest.raises(ValueError):
        EventRow(time,events,censored)


@pytest.mark.parametrize('initial,causes,rows', [(0,['A'],[EventRow(1,[1],0)]),(True,['A'],[EventRow(1,[1],0)]),(1,['A','A'],[EventRow(1,[1,0],0)]),(1,['bad label'],[EventRow(1,[1],0)]),(1,['A'],[]),(2,['A'],[EventRow(1,[1],0)]),(1,['A'],[EventRow(1,[2],0)]),(2,['A'],[EventRow(2,[1],0),EventRow(1,[1],0)]),(2,['A'],[EventRow(1,[1],0),EventRow(1,[1],0)]),(1,['A','B'],[EventRow(1,[1],0)])])
def test_invalid_tables(initial,causes,rows):
    with pytest.raises(ValueError):
        EventTable(initial,causes,rows)


@pytest.mark.parametrize('records', [[],[{'time':1,'cause':'unknown'}],[{'time':1}],[{'time':1,'cause':None,'extra':0}],[{'time':1.5,'cause':None}]])
def test_invalid_records(records):
    with pytest.raises(ValueError):
        from_records(records,['A'])
