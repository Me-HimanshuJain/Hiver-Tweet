"""Quick diagnostic on the NIM eval results."""
import json, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

with open('results/eval_report.json') as f:
    report = json.load(f)

errors   = report['n_errors']
n        = report['n_samples']
answered = n - errors

total_correct = round(report['accuracy'] * n)

print("=== DIAGNOSIS ===")
print(f"Total tweets        : {n}")
print(f"API errors (503)    : {errors}  ({errors/n:.1%}) — all retries exhausted, mapped to uncertain")
print(f"Uncertain (conf<0.5): {report['n_uncertain'] - errors}  — model expressed low confidence")
print(f"Successfully answered: {answered}")
print()
print(f"Reported accuracy (incl errors): {report['accuracy']:.1%}  ({total_correct}/{n})")
print(f"Corrected accuracy (excl errors): {total_correct}/{answered} = {total_correct/answered:.1%}")
print(f"Reported macro F1               : {report['macro_f1']:.3f}")
print()
print("=== WORST CONFUSION PAIRS ===")
for c in report['worst_confusions']:
    print(f"  {c['true']:35s} -> {c['predicted']:35s} ({c['count']}x)")
print()
print("=== PER-CLASS PERFORMANCE ===")
print(f"{'Intent':<35s} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Sup':>5}")
for intent, m in report['per_class_metrics'].items():
    print(f"{intent:<35s} {m['precision']:6.2f} {m['recall']:6.2f} {m['f1']:6.2f} {m['support']:5d}")
print()
print("=== CONFIDENCE CALIBRATION ===")
for c in report['confidence_calibration']:
    lo, hi = c['conf_range']
    print(f"  Q{c['quartile']} [{lo:.2f},{hi:.2f}]  n={c['n']:3d}  mean_conf={c['mean_confidence']:.3f}  acc={c['accuracy']:.1%}")
