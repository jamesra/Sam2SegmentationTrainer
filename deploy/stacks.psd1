# Sam2Trainer stack registry, read by connectomes-deploy's stack.ps1 (Import-PowerShellDataFile) through
# STACK_EXTRA_REGISTRIES (default entry SAM2TRAINER_REPO_ROOT_HOST:deploy/stacks.psd1).
#
# Same schema as Viking's deploy/stacks.psd1: ComposeFiles, ModelScript and the examples are relative to this
# repository's root, the host path in SAM2TRAINER_REPO_ROOT_HOST. The segmentation server moved here from Viking's
# Servers/SegmentationServer on 2026-11-20; the stack keeps its name so hosts' Run/viking-segmentation folders,
# models.json and env files are unchanged.
@{
    Stacks = @{
        'viking-segmentation' = @{
            Engine       = 'compose'
            RepoRootVar  = 'SAM2TRAINER_REPO_ROOT_HOST'
            ComposeFiles = @('SegmentationServer/docker-compose.yml')
            # The default model is the base service; every other model is a row of Run/viking-segmentation/models.json.
            Profiles     = @()
            # `stack.ps1 viking-segmentation model <prefix> <file.pt>` runs this script, which writes
            # Run/viking-segmentation/compose.generated.yaml (merged after ComposeFiles) and starts the servers.
            ModelScript  = 'SegmentationServer/rebuild-and-start.ps1'
            ModelsExample = 'deploy/examples/example.viking-segmentation.models.json'
            Required     = @('VIKING_HOSTNAME_SEGMENTATION', 'TLS_CERT_NAME')
            Example      = 'deploy/examples/example.viking-segmentation.run.env'
            Secrets      = @()
            # gRPC only (and the optional demo page); no plain HTTP endpoint to probe.
            HealthUrl    = ''
            Ports        = @(40443, 40444)
            Os           = @('linux', 'windows')
            Description  = 'SAM2 segmentation gRPC servers (GPU): the default model, plus one server per row of models.json'
        }
    }
}
