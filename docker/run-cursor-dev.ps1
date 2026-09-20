<#
.SYNOPSIS
  Run the SAM2 trainer cursor-dev Compose service (run-phase counterpart to docker-build.ps1).

.DESCRIPTION
  Starts docker/compose.cursor-dev.yaml service cursor-dev with the path-B CIFS override
  so //192.168.0.199/Data/Volumes is mounted at /storage4. This is separate from image
  build so NAS credentials and host bind paths never enter docker build.

  Requires docker/.env (hardlink or copy of D:\Docker\Run\sam2-dev\.env) for Compose
  substitution of NORNIR_NET_MOUNTS_DIR_HOST and NORNIR_NET_CREDS_DIR_HOST. Those dirs
  must exist (reuse Nornir's nornir-dev net-mounts; do not copy .cred files into git).

.PARAMETER RemainingArgs
  Passed to the container after the service name (for example an alternate command).
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$RemainingArgs
)

$ErrorActionPreference = 'Stop'

$DockerDir = $PSScriptRoot
$RepoRoot = Split-Path -Parent $DockerDir
$envFile = Join-Path $DockerDir '.env'
$exampleRel = 'docker/example.cursor-dev.run.env'

function Get-DotEnvValue {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Key
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    $prefix = "$Key="
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trim = $line.Trim()
        if ($trim.StartsWith($prefix)) {
            return $trim.Substring($prefix.Length).Trim()
        }
    }
    return $null
}

if (-not (Test-Path -LiteralPath $envFile)) {
    Write-Error @"
Missing $envFile (Compose substitution). Do:
  Copy $exampleRel to D:\Docker\Run\sam2-dev\.env
  New-Item -ItemType HardLink -Path docker\.env -Target D:\Docker\Run\sam2-dev\.env
"@
}

$mountsDir = Get-DotEnvValue -Path $envFile -Key 'NORNIR_NET_MOUNTS_DIR_HOST'
$credsDir = Get-DotEnvValue -Path $envFile -Key 'NORNIR_NET_CREDS_DIR_HOST'
if ([string]::IsNullOrWhiteSpace($mountsDir) -or [string]::IsNullOrWhiteSpace($credsDir)) {
    Write-Error "docker/.env must set NORNIR_NET_MOUNTS_DIR_HOST and NORNIR_NET_CREDS_DIR_HOST (see $exampleRel)."
}
if (-not (Test-Path -LiteralPath $mountsDir)) {
    Write-Error "NORNIR_NET_MOUNTS_DIR_HOST does not exist: $mountsDir"
}
if (-not (Test-Path -LiteralPath $credsDir)) {
    Write-Error "NORNIR_NET_CREDS_DIR_HOST does not exist: $credsDir"
}

$outputHost = Get-DotEnvValue -Path $envFile -Key 'SAM2_OUTPUT_HOST'
if ([string]::IsNullOrWhiteSpace($outputHost)) {
    $outputHost = 'D:/Docker/mounted-configs/sam2-trainer'
}
New-Item -ItemType Directory -Force -Path $outputHost | Out-Null

$composeFile = Join-Path $DockerDir 'compose.cursor-dev.yaml'
$composeFiles = @('-f', $composeFile)

$userRoot = $env:NORNIR_DOCKER_USER_ROOT
if ([string]::IsNullOrWhiteSpace($userRoot)) {
    $userRoot = 'D:\Docker'
}
$overrideFound = $false
foreach ($rel in @(
        'Run\sam2-dev\compose.net-mounts.override.yaml',
        'Run\nornir-dev\compose.net-mounts.override.yaml'
    )) {
    $override = Join-Path $userRoot $rel
    if (Test-Path -LiteralPath $override) {
        $composeFiles += @('-f', $override)
        Write-Host "Using net-mounts override: $override"
        $overrideFound = $true
        break
    }
}
if (-not $overrideFound) {
    Write-Error @"
Missing compose.net-mounts.override.yaml under $userRoot\Run\sam2-dev (preferred) or Run\nornir-dev.
Copy docker\example.compose.net-mounts.override.yaml to D:\Docker\Run\sam2-dev\compose.net-mounts.override.yaml
"@
}

Set-Location -LiteralPath $RepoRoot

$runArgs = @('compose') + $composeFiles + @('--env-file', $envFile, 'run', '--rm', 'cursor-dev')
foreach ($a in $RemainingArgs) {
    $runArgs += $a
}

Write-Host "docker $($runArgs -join ' ')"
& docker @runArgs
exit $LASTEXITCODE
