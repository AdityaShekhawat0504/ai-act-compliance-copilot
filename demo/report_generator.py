# report_generator.py
from jinja2 import Template
import pdfkit
import datetime

REPORT_TEMPLATE = """
<html>
<head><meta charset="utf-8"><title>Compliance Report</title></head>
<body>
<h1>AI Act Compliance Report</h1>
<p>Date: {{ date }}</p>
<h2>Risk Summary</h2>
<ul>
  <li>Risk Level: <b>{{ level }}</b> (score {{ score }})</li>
</ul>
<h3>Reasons</h3>
<ul>
{% for r in reasons %}
  <li>{{ r }}</li>
{% endfor %}
</ul>

<h2>Model Metrics</h2>
<ul>
  <li>Accuracy: {{ accuracy }}</li>
  <li>ROC AUC: {{ roc_auc }}</li>
</ul>

<h2>Fairness</h2>
<ul>
  <li>Statistical Parity Difference (male - female): {{ spd }}</li>
</ul>

<h2>Recommended Action Items</h2>
<ol>
{% for it in actions %}
  <li>{{ it }}</li>
{% endfor %}
</ol>

</body>
</html>
"""

def generate_pdf_report(out_path, level, score, reasons, metrics, spd, actions):
    html = Template(REPORT_TEMPLATE).render(
        date=datetime.datetime.utcnow().isoformat(timespec='seconds'),
        level=level,
        score=score,
        reasons=reasons,
        accuracy=round(metrics.get('accuracy', 0), 3),
        roc_auc=round(metrics.get('roc_auc', 0), 3),
        spd=spd,
        actions=actions
    )
    pdfkit.from_string(html, out_path)
    return out_path
