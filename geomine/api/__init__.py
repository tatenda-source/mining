"""GeoMine HTTP API -- product surface over the prediction and audit pipelines.

Run:
    uvicorn geomine.api.main:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                  service banner
    GET  /v1/health         liveness probe
    GET  /v1/benchmark      published benchmark numbers (verifiable)
    POST /v1/score          score a concession boundary against the GeoMine model
    POST /v1/audit          run the audit protocol on a customer-supplied dataset
"""
