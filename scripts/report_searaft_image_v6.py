#!/usr/bin/env python3
"""Read v6 SEA-RAFT A/B/C/D/E metrics_final.json into a publication-ready *candidate* CSV."""
from pathlib import Path
import csv
import json

MODES=('frozen','ft10','full','stage2-only','stage1-only')
SEQS=('indoor_flying1','indoor_flying2','indoor_flying3')
out=Path('results/searaft_image_v6_summary.csv')
out.parent.mkdir(parents=True,exist_ok=True)
rows=[]
for mode in MODES:
    file=Path(f'runs/searaft_image_v6_{mode}_seed0/metrics_final.json')
    if file.is_file():
        d=json.loads(file.read_text())
        rs=d.get('results',{})
        protocol=d.get('config',{}).get('data_protocol','UNKNOWN')
    else:
        rs={};protocol='MISSING'
    for seq in SEQS:
        m=rs.get(seq,{})
        rows.append({'Protocol':mode,'Sequence':seq,'EPE':m.get('EPE','MISSING'),
            'Outlier%':m.get('Outlier%', 'MISSING'),'1PE%':m.get('1PE%', 'MISSING'),
            '2PE%':m.get('2PE%', 'MISSING'),'3PE%':m.get('3PE%', 'MISSING'),
            'Dataset_protocol':protocol,'Source':str(file) if file.exists() else 'MISSING'})
with out.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Protocol                   flying1 EPE   flying2 EPE   flying3 EPE')
for mode in MODES:
    items=[r['EPE'] for r in rows if r['Protocol']==mode]
    fmt=lambda x: f'{x:.4f}' if isinstance(x,(int,float)) else str(x)
    print(f'{mode:25s} {fmt(items[0]):>12s} {fmt(items[1]):>13s} {fmt(items[2]):>13s}')
print('CSV:',out)
print('NOT an exact ScaleEvent Table 6 reproduction until sampling/filtering/decoder equivalence verified.')
