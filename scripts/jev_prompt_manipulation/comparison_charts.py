"""Article bar plots, derived from recorded scores. Zero-based labelled scales.
Percent plots always span 0–100%; the single-review plot spans its 0–4 rubric.
Different campaigns remain separate, with each numerator/denominator visible.
"""
import html
import json
from pathlib import Path


def bar(value, maximum, label):
    assert 0 <= value <= maximum
    return f'<div class="bar-measure"><span class="bar-track" aria-hidden="true"><span class="bar-fill" style="width:{100*value/maximum:.8f}%"></span></span><span class="bar-value">{label}</span></div>'


def figure(chart_id, title, subtitle, rows, note, maximum=100):
    suffix='%' if maximum==100 else ''
    axis='<div class="bar-axis" aria-hidden="true">'+''.join(f'<span>{maximum*i/4:g}{suffix}</span>' for i in range(5))+'</div>'
    body=''.join(f'<div class="bar-row"><p class="bar-model">{html.escape(name)}</p>'+bar(value,maximum,label)+'</div>' for name,value,label in rows)
    return f'<figure class="comparison-plot" aria-labelledby="{chart_id}"><figcaption><h3 id="{chart_id}">{title}</h3><p>{subtitle}</p></figcaption>{axis}{body}<p class="plot-note">{note}</p></figure>'


def rate_row(label,n,d):
    rate=100*n/d
    value=f'{rate:.2f}' if 0<rate<1 else f'{rate:.1f}'
    return label,rate,f'{value}% <span>({n:,}/{d:,})</span>'


def render_hardening(data: Path):
    rows=[json.loads(line) for line in (data/'jev-heldout.jsonl').read_text().splitlines()]
    clean={(r['item']['id'],r['defense'],r['repeat']):r for r in rows if r['attack']['name']=='clean'}
    values=[]
    for policy,label in [('basic','Basic rules'),('hardened','Ignore embedded commands'),('strict','Also check every part')]:
        attacks=[r for r in rows if r['defense']==policy and r['attack']['name'] not in ('clean','neutral') and clean[r['item']['id'],policy,r['repeat']]['choice']==r['item']['expected']]
        values.append(rate_row(label,sum(r['choice']!=r['item']['expected'] for r in attacks),len(attacks)))
    return figure('hardening-chart-title','Clearer rules stopped the first set of attacks','Jev · same eight held-out messages and selected attacks · lower is better.',values,'Wrong changes / attacks with a correct original answer. Each attack was run twice. The rules change here; the messages and attacks do not. <a href="#results">First-test records</a>.')


def render_larger(data: Path):
    scores=json.loads((data/'comprehensive/scores.json').read_text())
    values=[]
    for run,label in [('static-jev','Larger fixed test'),('transfer-jev','Selected transfer attacks')]:
        r=scores[run]['overall'];values.append(rate_row(label,r['flips'],r['eligible_attacks']))
    return figure('larger-chart-title','Strict rules still allowed failures','Jev · two separate follow-up tests · lower is better.',values,'These are different attack sets, not a before/after comparison. One repeat each. Tiny bars still represent observed failures: 1/1,056 and 2/72. <a href="#followup-evidence">Follow-up records</a>.')


def render_review(before,after):
    return figure('review-chart-title','The same five-star review, a different score','Laya English · one recorded example · expected score: 4.',[
        ('Original review',before,f'{before:.2f} <span>/ 4</span>'),
        ('With the instruction',after,f'{after:.2f} <span>/ 4</span>')],
        'The bars show returned scores, not attack success rates. On this rubric, 0 means one star and 4 means five stars. <a href="/data/demos/jev-prompt-manipulation/laya-typed/examples.json">Exact input and output</a>.',maximum=4)


def render_comparison(data: Path):
    original=json.loads((data/'comprehensive/scores.json').read_text())
    extension=json.loads((data/'laya-typed/scores.json').read_text())
    configs=[('jev','Jev 1.13.0'),('winnow','Winnow-12B Q8'),('semif','SemIf · Qwen3.5-4B'),('laya-multilingual','Laya multilingual'),('laya','Laya English')]
    reports=[(label,(original if key in ('jev','winnow','semif') else extension)['static-'+key]) for key,label in configs]
    assert len({r['plan_sha256'] for _,r in reports})==1
    assert all(r['complete'] and r['overall']['planned_calls']==2160 for _,r in reports)
    values=[rate_row(label,r['overall']['flips'],r['overall']['eligible_attacks']) for label,r in reports]
    return '<div id="model-comparison"><p>I then compared all five configurations on the larger fixed test: the same 24 texts, 44 attacks per text, and strict rules. This included Laya’s English and multilingual checkpoints. These are Jev-like decision systems, but they do not all share Jev’s architecture.</p>'+figure('attack-chart-title','How often did an attack change a correct answer?','Larger fixed test · wrong changes / attacks with a correct original answer · lower is better.',values,'Not a like-for-like ranking. Laya English is not intended for this three-language set, and both Laya checkpoints already miss 11–12 of the 24 texts before any attack, which shrinks their denominators. One repeat, no confidence intervals. All bars use a 0–100% scale; Jev’s 1/1,056 is small, not zero. Eight scenario groups in three languages.')+'''<p>Jev had the fewest wrong changes on this set. Laya’s denominators are smaller because it already missed 11 or 12 of the 24 original messages; those cases do not count as attack successes. Its harmless-padding controls also failed often. The English checkpoint is not intended for the full three-language set. Those details remain in the appendix.</p>
<p>So Jev had the fewest observed failures on this particular set—and that still included a scam getting through. These results compare the tested inputs and settings, not real-world failure rates.</p>
<p class="plot-source"><a href="/data/demos/jev-prompt-manipulation/comprehensive/scores.json">Jev, SemIf, and Winnow scores</a> · <a href="/data/demos/jev-prompt-manipulation/laya-typed/scores.json">Laya scores</a> · <a href="#followup-evidence">Full evidence and control results</a>.</p></div>'''
