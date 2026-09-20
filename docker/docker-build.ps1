<#
.SYNOPSIS
  Build the sam2-trainer:dev image (build-phase counterpart to run-cursor-dev.ps1).

.DESCRIPTION
  Builds docker/Dockerfile from the Sam2SegmentationTrainer repository root and tags
  sam2-trainer:dev. Separating this from run-cursor-dev.ps1 keeps image build-args
  out of the Compose/CIFS run path.

  Merge optional KEY=VALUE files from the invocation directory (not the repo):
    build.env
    .build.sam2-trainer-dev.env
  Later files override earlier keys. Then --build-arg on the command line from those keys.
  Committed docker/example.sam2-trainer-dev.build.env is a template only and is not read.

  Invoke from D:\Docker\Builds\sam2-trainer so machine-local build.env files stay out of git.

.PARAMETER NoCache
  Pass --no-cache to docker build.

.PARAMETER ExtraArgs
  Additional docker build arguments appended after the merged --build-arg list.
#>
param(
    [switch]$NoCache,

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = 'Stop'

function Read-DotEnvFile {
    param([Parameter(Mandatory)][string]$Path)
    $map = [ordered]@{}
    if (-not (Test-Path -LiteralPath $Path)) {
        return $map
    }
    Write-Host "Found and merged: $Path"
    Get-Content -LiteralPath $Path | ForEach-Object {
        $line = $_.Trim()
        if ($line -eq '' -or $line.StartsWith('#')) {
            return
        }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) {
            return
        }
        $key = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim()
        $map[$key] = $value
    }
    return $map
}

$ImageTag = 'sam2-trainer:dev'
$ImageId = $ImageTag -replace ':', '-'
$BuildEnvRoot = (Get-Location).ProviderPath
$DockerDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $DockerDir

Get-ChildItem -LiteralPath $BuildEnvRoot -Filter 'example.*.build.env' -ErrorAction SilentlyContinue |
    ForEach-Object {
        Write-Host ("Template name not merged: {0}`n  Copy to {1}" -f $_.FullName, (Join-Path $BuildEnvRoot ".build.$ImageId.env"))
    }

$merged = [ordered]@{}
$mergeNames = @('build.env', ".build.$ImageId.env")
# Older layout used the image name without the :dev suffix.
if (Test-Path -LiteralPath (Join-Path $BuildEnvRoot '.build.sam2-trainer.env')) {
    $mergeNames += '.build.sam2-trainer.env'
}
foreach ($name in $mergeNames) {
    $path = Join-Path $BuildEnvRoot $name
    if (Test-Path -LiteralPath $path) {
        $part = Read-DotEnvFile -Path $path
        foreach ($k in $part.Keys) {
            $merged[$k] = $part[$k]
        }
    }
    else {
        Write-Host "Not found (skipped): $path"
    }
}

# Relative -f and context after cd. An absolute Windows path (D:\...) is parsed by
# buildx as a URL scheme "D:", which drops the context. Skip empty ExtraArgs: a
# null splat from ./build.ps1 becomes "" and that empty PATH triggers
# "'docker buildx build' requires 1 argument".
$buildArgs = @(
    'build',
    '-f', 'docker/Dockerfile',
    '-t', $ImageTag
)
if ($NoCache) {
    $buildArgs += '--no-cache'
}
foreach ($k in $merged.Keys) {
    $buildArgs += @('--build-arg', "$k=$($merged[$k])")
}
if ($null -ne $ExtraArgs) {
    foreach ($a in $ExtraArgs) {
        if ([string]::IsNullOrWhiteSpace($a)) {
            continue
        }
        $buildArgs += $a
    }
}
$buildArgs += @('--', '.')

$dockerCmd = Get-Command docker.exe -CommandType Application -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $dockerCmd) {
    $dockerCmd = Get-Command docker -CommandType Application -ErrorAction Stop |
        Select-Object -First 1
}

Write-Host "Building ${ImageTag} (context $RepoRoot, env from $BuildEnvRoot)"
Write-Host "$($dockerCmd.Source) $($buildArgs -join ' ')"
Push-Location -LiteralPath $RepoRoot
try {
    & $dockerCmd.Source @buildArgs
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
