param(
    [ValidateSet("unit", "integration", "e2e", "e2e-slow", "all", "changed")]
    [string]$Layer = "unit"
)

switch ($Layer) {
    "unit"        { py -m pytest tests/unit/ -v --cov=src --cov-branch --cov-report=term-missing --cov-fail-under=80 }
    "integration" { py -m pytest tests/integration/ -v --timeout=120 }
    "e2e"         { py -m pytest tests/e2e/ -v --timeout=60 -m "not slow" }
    "e2e-slow"    { py -m pytest tests/e2e/ -v --timeout=300 -m "slow" }
    "all"         { py -m pytest tests/unit/ tests/integration/ tests/e2e/ -v -m "not slow" }
    "changed"     { py -m pytest tests/unit/ --testmon -q }
}
