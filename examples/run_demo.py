"""Exact synthetic counting tables and explicit censoring counterexamples."""

import argparse
from fractions import Fraction as F
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from competing_events import EventRow, EventTable, estimate, from_records
from competing_events.reference import matrix_reference


def hazard_example(competing_hazard):
    remaining, rows = 100000, []
    for time in range(1,6):
        counts = [remaining*F(1,10), remaining*competing_hazard]
        assert all(v.denominator == 1 for v in counts)
        events = [int(v) for v in counts]
        next_remaining = remaining-sum(events)
        rows.append(EventRow(time,events,next_remaining if time == 5 else 0))
        remaining = next_remaining
    return estimate(EventTable(100000,['A','B'],rows))


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    competing_hazards = [F(0),F(1,10),F(3,10),F(3,5)]
    hazard_curves = {str(h):hazard_example(h) for h in competing_hazards}
    complete = estimate(EventTable(1000,['A','B'],[EventRow(1,[200,300],0),EventRow(2,[100,150],250)]))
    proportional = estimate(EventTable(1000,['A','B'],[EventRow(0,[0,0],300),EventRow(1,[140,210],0),EventRow(2,[70,105],175)]))
    world_a = ([{'time':1,'cause':'A'} for _ in range(200)]+[{'time':1,'cause':'B'} for _ in range(300)]
               +[{'time':2,'cause':'A'} for _ in range(100)]+[{'time':2,'cause':'B'} for _ in range(150)]
               +[{'time':2,'cause':None} for _ in range(250)])
    hidden = {i for i,row in enumerate(world_a) if row['cause'] == 'A'}
    world_b = [{'time':2,'cause':None} if i in hidden else row.copy() for i,row in enumerate(world_a)]
    observe = lambda rows: [{'time':0,'cause':None} if i in hidden else row.copy() for i,row in enumerate(rows)]
    observed_a, observed_b = observe(world_a), observe(world_b)
    assert observed_a == observed_b
    informative = estimate(from_records(observed_a,['A','B']))
    latent_b = estimate(from_records(world_b,['A','B']))
    assert estimate(from_records(world_a,['A','B'])).at(2) == complete.at(2)
    assert complete.at(2) == proportional.at(2)
    tied = estimate(EventTable(10,['A','B'],[EventRow(1,[2,3],5)]))
    reference_records = [{'time':1,'cause':'A'}]*2+[{'time':1,'cause':'B'}]*3+[{'time':1,'cause':None}]*2+[{'time':2,'cause':'A'}]+[{'time':2,'cause':None}]*2
    reference_curve = estimate(from_records(reference_records,['A','B']))
    direct = matrix_reference(reference_records,['A','B'])
    assert [(p.time,(p.survival,*p.incidence)) for p in reference_curve.points] == direct
    summary = {
        'data':'Deterministic synthetic integer counts; no sampled or real observations.',
        'constant_hazards':{'target_hazard':'1/10','horizon':5,'curves':{h:c.to_dict() for h,c in hazard_curves.items()}},
        'censoring':{'complete':complete.to_dict(),'proportional':proportional.to_dict(),'informative':informative.to_dict(),
                     'proportional_description':'Exactly 30% removed at time zero within each latent outcome category; a balanced deterministic construction, not a random-censoring simulation.'},
        'nonidentification':{'same_observed_individual_records':observed_a == observed_b,
                             'latent_world_A':complete.to_dict(),'latent_world_B':latent_b.to_dict(),
                             'hidden_record_count':len(hidden),'observed':informative.to_dict()},
        'ties':{'event_before_censor':tied.to_dict(),'censor_first_alternative_incidence':['2/5','3/5'],
                'interpretation':'The alternative changes the risk-set convention; coarsened timestamps alone cannot determine the true ordering.'},
        'matrix_reference':{'matches_exactly':True,'curve':reference_curve.to_dict(),
                            'states':[{'time':t,'state':list(map(str,s))} for t,s in direct]},
        'scope':'Exact point calculations do not provide confidence intervals or repair informative censoring.'
    }
    (output/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    plot_rows = []
    for h,c in hazard_curves.items():
        for p in c.points:
            plot_rows.append({'competing_hazard':h,'time':p.time,'target_incidence':str(p.incidence[0]),'target_net_risk':str(p.net_risk[0]),'survival':str(p.survival)})
    (output/'hazard_curves.json').write_text(json.dumps(plot_rows,indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes = plt.subplots(2,2,figsize=(12,8.4),layout='constrained')
    fig.suptitle('Competing events: incidence, censoring, and what data cannot identify',fontsize=16,fontweight='bold')
    blue,orange,purple = '#176885','#b45822','#7660a4'
    c = hazard_curves['3/10']
    times = [0]+[p.time for p in c.points]
    ax=axes[0,0]
    ax.step(times,[0]+[float(p.incidence[0]) for p in c.points],where='post',lw=2,color=blue,label='Cumulative incidence of A')
    ax.step(times,[0]+[float(p.net_risk[0]) for p in c.points],where='post',lw=2,color=orange,label='1 − cause-specific KM')
    ax.set(xlabel='Discrete time',ylabel='Probability functional',title='A. Target hazard 0.1; competing hazard 0.3',ylim=(0,.48))
    ax.legend(fontsize=9,loc='upper left')
    ax=axes[0,1]
    ax.bar(range(4),[float(hazard_curves[str(h)].points[-1].incidence[0]) for h in competing_hazards],color=blue)
    ax.axhline(float(c.points[-1].net_risk[0]),color=orange,ls='--',label='Same net risk: 0.40951')
    ax.set(xticks=range(4),xticklabels=list(map(str,map(float,competing_hazards))),xlabel='Competing event hazard',ylabel='Incidence of A by time 5',title='B. Same target hazard, different incidence',ylim=(0,.52))
    ax.legend(fontsize=9,loc='upper right')
    ax=axes[1,0]
    curves=[complete,proportional,informative]
    ax.bar(range(3),[float(v.points[-1].incidence[0]) for v in curves],color=[blue,blue,orange])
    ax.axhline(.3,color='#555555',ls='--',label='Complete-cohort truth: 0.30')
    ax.set(xticks=range(3),xticklabels=['No early\ncensoring','Proportional\ncensoring','All future A\nhidden at time 0'],ylabel='Estimated incidence of A by time 2',title='C. Aalen–Johansen cannot fix informative censoring',ylim=(0,.43))
    ax.legend(fontsize=9,loc='upper right')
    ax=axes[1,1]
    ax.bar([0,1],[.3,0],color=purple)
    ax.text(1,.015,'0',ha='center',va='bottom',color=purple)
    ax.set(xticks=[0,1],xticklabels=['Latent world A','Latent world B'],ylabel='True incidence of A by time 2',title='D. Identical observed records, different truths',ylim=(0,.43))
    ax.text(.5,.37,'300 records censored at time 0 in both worlds',ha='center',fontsize=9)
    fig.savefig(output/'incidence_audit.png',dpi=160)
    plt.close(fig)
    print(json.dumps({'hazard_scenarios':{h:{'incidence_A':str(v.points[-1].incidence[0]),'net_risk_A':str(v.points[-1].net_risk[0])} for h,v in hazard_curves.items()},
                      'complete_incidence_A':str(complete.points[-1].incidence[0]),'proportional_incidence_A':str(proportional.points[-1].incidence[0]),
                      'informative_incidence_A':str(informative.points[-1].incidence[0]),'same_observed_records':observed_a==observed_b},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('results'))
    run(parser.parse_args().output)
