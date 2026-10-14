"""Image-level FIRE candidate metrics; failures are abstentions, never negatives."""
import math


def ratio(a, b):
    return a / b if b else None


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def summarize(rows):
    reviewed = [r for r in rows if r['expectedFire'] is not None]
    scored = [r for r in reviewed if r['status'] == 'OK']
    tp = sum(r['expectedFire'] and r['predictedFire'] for r in scored)
    fp = sum(not r['expectedFire'] and r['predictedFire'] for r in scored)
    fn = sum(r['expectedFire'] and not r['predictedFire'] for r in scored)
    tn = sum(not r['expectedFire'] and not r['predictedFire'] for r in scored)
    missed = fn + sum(r['expectedFire'] for r in reviewed if r['status'] != 'OK')
    times = [r['latencyMs'] for r in rows]
    return dict(images=len(rows), reviewed=len(reviewed), pendingReview=len(rows)-len(reviewed),
                scored=len(scored), reviewedAbstentions=len(reviewed)-len(scored),
                truePositives=tp, falsePositives=fp, falseNegatives=fn, trueNegatives=tn,
                accuracy=ratio(tp+tn, len(scored)), precision=ratio(tp, tp+fp), recall=ratio(tp, tp+fn),
                f1=ratio(2*tp, 2*tp+fp+fn), falsePositiveRate=ratio(fp, fp+tn),
                falseAlertsPer100ScoredImages=100*fp/len(scored) if scored else None,
                missedPositiveImagesIncludingAbstentions=missed,
                endToEndRecall=ratio(tp, sum(r['expectedFire'] for r in reviewed)),
                unusableRate=ratio(sum(r['status']=='UNUSABLE' for r in rows), len(rows)),
                errorRate=ratio(sum(r['status']=='ERROR' for r in rows), len(rows)),
                latencyMs=dict(mean=ratio(sum(times), len(times)), p50=percentile(times,.5),
                               p95=percentile(times,.95), maximum=max(times) if times else None),
                mAP50=None, mAP50_95=None, falseAlertsPerCameraHour=None)
