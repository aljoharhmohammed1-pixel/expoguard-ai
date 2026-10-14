"""Deterministic fact summary; supplied grouping is not incident confirmation."""
from pydantic import Field, model_validator
from ai.core.analysis import Contract, Identifier
from ai.operations.correlation import AlertFacts

LABELS = {'FIRE':'مؤشرات حريق محتمل', 'CROWD':'مؤشرات ازدحام', 'MEDICAL':'بلاغ طبي',
          'TECHNICAL_FAILURE':'بلاغ عطل تقني', 'OTHER':'بلاغ آخر', 'ACCESSIBILITY':'بلاغ إتاحة',
          'CLEANLINESS':'بلاغ نظافة', 'SECURITY':'بلاغ أمني'}


class SummaryRequest(Contract):
    alerts: list[AlertFacts] = Field(min_length=1, max_length=100)

    @model_validator(mode='after')
    def unique(self):
        if len({a.alertId for a in self.alerts}) != len(self.alerts):
            raise ValueError('Duplicate alert IDs')
        return self


class SummaryResult(Contract):
    summaryAr: Identifier
    sourceCount: int = Field(ge=1)
    alertIds: list[Identifier]
    policyVersion: str = 'facts-summary-v0.1.0'
    limitations: list[str]


def summarize(payload):
    payload = SummaryRequest.model_validate(payload)
    alerts = sorted(payload.alerts, key=lambda a:(a.observedAt,a.alertId))
    categories = '، '.join(LABELS[c] for c in sorted({a.category for a in alerts}))
    zones = sorted({a.zoneId for a in alerts if a.zoneId is not None})
    location = 'المناطق المزوّدة: '+ '، '.join(zones)+'.' if zones else 'لم تُزوّد منطقة موثوقة.'
    text = f'ورد {len(alerts)} بلاغًا يتضمن: {categories}. {location} لا يؤكد هذا الملخص وقوع حادث واحد؛ يلزم التحقق البشري.'
    return SummaryResult(summaryAr=text, sourceCount=len(alerts), alertIds=[a.alertId for a in alerts],
        limitations=['Source count means distinct supplied alert records, not independent cameras or witnesses.',
                     'Grouping is supplied by the caller and is not confirmed by this summary.'])
