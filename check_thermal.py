import json
from collections import Counter

with open('thermal/InfraredSolarModules/module_metadata.json') as f:
    data = json.load(f)

# Count original classes
orig_counts = Counter(v['anomaly_class'] for v in data.values())

# Your 12 → 8 remapping
THERMAL_REMAP = {
    'No-Anomaly':     0,
    'Soiling':        1,
    'Shadowing':      2,
    'Vegetation':     2,
    'Hot-Spot':       4,   # thin film hotspot → internal
    'Hot-Spot-Multi': 4,   # thin film multi hotspot → internal
    'Diode':          4,   # bypass diode → internal
    'Diode-Multi':    4,   # multi bypass diode → internal
    'Cell':           5,   # crystalline cell failure
    'Cell-Multi':     5,   # crystalline multi-cell failure
    'Cracking':       6,
    'Offline-Module': 7,
}

CLASS_MAP = {
    0: 'No Anomaly',
    1: 'Soiling/Pollution',
    2: 'Shadowing/Vegetation',
    3: 'Hotspot-Surface',
    4: 'Hotspot-Internal',
    5: 'Cell/String Failure',
    6: 'Structural Damage',
    7: 'Offline Module',
}

print('Original 12-class distribution:')
print('-' * 45)
remapped = Counter()
for orig_label, cnt in sorted(orig_counts.items(), key=lambda x: -x[1]):
    remap = THERMAL_REMAP.get(orig_label, 'UNMAPPED')
    your_class = CLASS_MAP.get(remap, '3 or 4 (hotspot check needed)') if remap != '3or4' else 'Class 3 or 4 (hotspot check needed)'
    print(f'  {orig_label:<20} {cnt:>5}  →  {your_class}')
    if isinstance(remap, int):
        remapped[remap] += cnt

print()
print('Remapped 8-class distribution (thermal only):')
print('-' * 45)
for cls_id in range(8):
    cnt = remapped.get(cls_id, 0)
    status = '✅' if cnt > 100 else ('⚠️ ' if cnt > 0 else '❌')
    print(f'  Class {cls_id} ({CLASS_MAP[cls_id]:<25}) {cnt:>5}  {status}')

print(f'\n  Total accounted for: {sum(remapped.values())} / {len(data)}')
print(f'  Hot-Spot entries needing soiling check: {orig_counts.get("Hot-Spot", 0)}')