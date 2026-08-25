param(
    [ValidateSet('plan','smoke','fetch','validate','verify','compute','classify','report','run')]
    [string]$Command = 'plan',
    [string]$Config = 'configs/task0.yaml'
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location -LiteralPath $ProjectRoot
try {
    python -m jhtdb_regimes.cli $Command $Config
    if ($LASTEXITCODE -ne 0) {
        throw "jhtdb-regimes failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
