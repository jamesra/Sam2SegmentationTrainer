<#
.SYNOPSIS
  Shared helpers for run-cursor-dev.ps1 and docker-build.ps1 (dot-source; do not run).

.DESCRIPTION
  The trainer stack is the same for TEM and SEM. What differs (SAM2_MODALITY,
  SAM2_LOCAL_DATA_HOST, SAM2_OUTPUT_HOST, SAM2_CHECKPOINT_ROOT) lives in TEM.env / SEM.env,
  layered over the shared docker/.env: compose gets  --env-file docker/.env --env-file <Modality>.env.
  Only one modality runs at a time because both use the same ports and project name.
#>

$script:Sam2ComposeProject = 'sam2-trainer'
$script:Sam2Modalities = @('TEM', 'SEM')

function Get-Sam2DotEnvValue {
    param(
        [Parameter(Mandatory)][string]$Path,
        [Parameter(Mandatory)][string]$Key
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    $prefix = "$Key="
    $value = $null
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trim = $line.Trim()
        if ($trim.StartsWith($prefix)) {
            $value = $trim.Substring($prefix.Length).Trim()   # last assignment wins, as in compose
        }
    }
    return $value
}

# $NORNIR_DOCKER_USER_ROOT\Run\sam2-dev\<Modality>.env (default root D:\Docker), else docker\<Modality>.env.
function Resolve-Sam2ModalityEnvFile {
    param(
        [Parameter(Mandatory)][string]$Modality,
        [Parameter(Mandatory)][string]$DockerDir
    )
    $userRoot = $env:NORNIR_DOCKER_USER_ROOT
    if ([string]::IsNullOrWhiteSpace($userRoot)) { $userRoot = 'D:\Docker' }
    $candidates = @(
        (Join-Path $userRoot "Run\sam2-dev\$Modality.env"),
        (Join-Path $DockerDir "$Modality.env")
    )
    foreach ($c in $candidates) {
        if (Test-Path -LiteralPath $c) { return $c }
    }
    Write-Error @"
Missing $Modality.env. Looked in:
  $($candidates -join "`n  ")
Copy docker\example.$Modality.env to $($candidates[0]) and adjust the paths.
"@
}

# SAM2_MODALITY of each running container of the sam2-trainer project (empty list when none).
# -Service limits it to one compose service (e.g. cursor-dev).
function Get-Sam2RunningModalities {
    param(
        [Parameter(Mandatory)][string]$DockerExe,
        [string]$Service = ''
    )
    $filters = @('--filter', "label=com.docker.compose.project=$script:Sam2ComposeProject")
    if ($Service) { $filters += @('--filter', "label=com.docker.compose.service=$Service") }
    $ids = & $DockerExe ps -q @filters
    $found = New-Object System.Collections.Generic.List[object]
    foreach ($id in @($ids)) {
        if ([string]::IsNullOrWhiteSpace($id)) { continue }
        # A container that docker ps lists but inspect cannot read is stale daemon state, not a trainer.
        $nameRaw = & $DockerExe inspect $id --format '{{.Name}}' 2>$null
        if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($nameRaw)) {
            Write-Warning "Ignoring container ${id}: docker ps lists it but docker inspect fails (stale Docker state; docker rm -f ${id} clears it)."
            continue
        }
        $name = ([string]$nameRaw).Trim().TrimStart('/')
        $envLines = & $DockerExe inspect $id --format '{{range .Config.Env}}{{println .}}{{end}}' 2>$null
        $mod = ''
        foreach ($l in @($envLines)) {
            if ($l -like 'SAM2_MODALITY=*') { $mod = $l.Substring('SAM2_MODALITY='.Length).Trim() }
        }
        $imageId = ([string](& $DockerExe inspect $id --format '{{.Image}}' 2>$null)).Trim()
        $found.Add([pscustomobject]@{ Id = $id; Name = $name; Modality = $mod; ImageId = $imageId })
    }
    # No leading comma: that would wrap an empty result in an outer array, so callers would see one phantom entry.
    return $found.ToArray()
}

# When a trainer (cursor-dev) of a different modality is running, say so and ask whether to replace
# it with the requested one; "No" (or a non-interactive session) stops the script. The two modalities
# share the project and ports, so they cannot run side by side, and replacing interrupts any run in
# progress. A container with no SAM2_MODALITY predates this scheme and is treated as TEM. The gallery
# is not checked: it is not a run, and switching only needs it recreated (the scripts print the command).
# -StopOnConfirm also stops the old container now (run-cursor-dev.ps1 starts a separate one-off
# container); without it the caller recreates the service itself (docker-build.ps1).
function Confirm-Sam2ReplaceOtherModality {
    param(
        [Parameter(Mandatory)][string]$DockerExe,
        [Parameter(Mandatory)][string]$Modality,
        [switch]$StopOnConfirm
    )
    $others = @(Get-Sam2RunningModalities -DockerExe $DockerExe -Service 'cursor-dev' | ForEach-Object {
            $running = if ($_.Modality) { $_.Modality } else { 'TEM' }
            if ($running -ne $Modality) { [pscustomobject]@{ Id = $_.Id; Name = $_.Name; Running = $running } }
        })
    if ($others.Count -eq 0) { return }

    $names = ($others | ForEach-Object { "$($_.Name) ($($_.Running))" }) -join ', '
    Write-Warning "A $($others[0].Running) trainer is running: $names. You asked for $Modality."
    $question = "Replace it with ${Modality}? That stops the running container and interrupts any training in it."
    try {
        $choice = $Host.UI.PromptForChoice('Modality mismatch', $question, @('&Yes, replace it', '&No, leave it running'), 1)
    }
    catch {
        Write-Error "Cannot prompt in a non-interactive session. Stop $names yourself (docker stop ...) or run this interactively."
    }
    if ($choice -ne 0) {
        Write-Error "Left $names running. Nothing was changed."
    }
    if (-not $StopOnConfirm) { return }
    foreach ($c in $others) {
        Write-Host "Stopping $($c.Name) ..."
        & $DockerExe stop -t 30 $c.Id | Out-Null
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "docker stop failed; trying docker rm -f (the container is removed, bind-mounted data is not touched)."
            & $DockerExe rm -f $c.Id | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Docker could not stop $($c.Name) (its state is stale). Restart Docker Desktop, then run this again."
            }
        }
    }
}

# The pretrained checkpoint (~0.9 GB) is downloaded per output folder. Reuse another modality's copy.
function Copy-Sam2PretrainedCheckpoint {
    param(
        [Parameter(Mandatory)][string]$Modality,
        [Parameter(Mandatory)][string]$DockerDir,
        [Parameter(Mandatory)][string]$OutputHost
    )
    $name = 'sam2.1_hiera_large.pt'
    $dest = Join-Path $OutputHost "checkpoints\$name"
    if (Test-Path -LiteralPath $dest) { return }
    foreach ($other in ($script:Sam2Modalities | Where-Object { $_ -ne $Modality })) {
        try { $otherEnv = Resolve-Sam2ModalityEnvFile -Modality $other -DockerDir $DockerDir } catch { continue }
        $otherOut = Get-Sam2DotEnvValue -Path $otherEnv -Key 'SAM2_OUTPUT_HOST'
        if ([string]::IsNullOrWhiteSpace($otherOut)) { continue }
        $src = Join-Path $otherOut "checkpoints\$name"
        if (Test-Path -LiteralPath $src) {
            New-Item -ItemType Directory -Force -Path (Split-Path -Parent $dest) | Out-Null
            Copy-Item -LiteralPath $src -Destination $dest
            Write-Host "Copied pretrained checkpoint from the $other output folder."
            return
        }
    }
}

# Validate <Modality>.env, check that the data mirror has a `current` link, create the output
# folder and reuse the pretrained checkpoint. Returns the path of <Modality>.env.
function Initialize-Sam2Modality {
    param(
        [Parameter(Mandatory)][string]$Modality,
        [Parameter(Mandatory)][string]$DockerDir
    )
    $modalityEnv = Resolve-Sam2ModalityEnvFile -Modality $Modality -DockerDir $DockerDir
    Write-Host "Modality $Modality (env: $modalityEnv)"
    foreach ($key in 'SAM2_MODALITY', 'SAM2_LOCAL_DATA_HOST', 'SAM2_OUTPUT_HOST', 'SAM2_CHECKPOINT_ROOT') {
        if ([string]::IsNullOrWhiteSpace((Get-Sam2DotEnvValue -Path $modalityEnv -Key $key))) {
            Write-Error "$modalityEnv must set $key (see docker\example.$Modality.env)."
        }
    }
    if ((Get-Sam2DotEnvValue -Path $modalityEnv -Key 'SAM2_MODALITY') -ne $Modality) {
        Write-Error "$modalityEnv sets SAM2_MODALITY to something other than $Modality."
    }

    $dataHost = Get-Sam2DotEnvValue -Path $modalityEnv -Key 'SAM2_LOCAL_DATA_HOST'
    if (-not (Test-Path -LiteralPath (Join-Path $dataHost 'current'))) {
        $syncScript = if ($Modality -eq 'SEM') { 'scripts/WSL/sem-sync_training_data.sh' } else { 'scripts/WSL/sync_training_data.sh' }
        Write-Error "No training data selected for ${Modality}: $dataHost\current is missing. In WSL run $syncScript <version>."
    }

    $outputHost = Get-Sam2DotEnvValue -Path $modalityEnv -Key 'SAM2_OUTPUT_HOST'
    New-Item -ItemType Directory -Force -Path $outputHost | Out-Null
    Copy-Sam2PretrainedCheckpoint -Modality $Modality -DockerDir $DockerDir -OutputHost $outputHost
    return $modalityEnv
}

# gallery.codepharm.net redirects to the gallery of the modality the trainer runs. gallery-router reads that from
# active.conf in SAM2_GALLERY_ROUTER_DIR (the shared .env; default D:\Docker\mounted-configs\sam2-trainer\gallery-router).
function Set-Sam2ActiveGalleryModality {
    param(
        [Parameter(Mandatory)][string]$DockerExe,
        [Parameter(Mandatory)][ValidateSet('TEM', 'SEM')][string]$Modality,
        [string]$EnvFile = ''
    )
    $dir = if ($EnvFile) { Get-Sam2DotEnvValue -Path $EnvFile -Key 'SAM2_GALLERY_ROUTER_DIR' } else { $null }
    if ([string]::IsNullOrWhiteSpace($dir)) { $dir = 'D:\Docker\mounted-configs\sam2-trainer\gallery-router' }
    New-Item -ItemType Directory -Force -Path $dir | Out-Null
    $line = 'set $gallery_target ' + $Modality.ToLowerInvariant() + ';'
    [System.IO.File]::WriteAllText((Join-Path $dir 'active.conf'), $line + "`n")
    $running = & $DockerExe ps -q --filter 'name=^gallery-router$'
    if ($running) {
        & $DockerExe exec gallery-router nginx -s reload | Out-Null
        Write-Host "gallery.codepharm.net now redirects to $($Modality.ToLowerInvariant()).gallery.codepharm.net"
    }
    else {
        Write-Host "gallery-router is not running; gallery.codepharm.net will redirect to $($Modality.ToLowerInvariant()) once it starts."
    }
}
