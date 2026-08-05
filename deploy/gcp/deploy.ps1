<#
.SYNOPSIS
    Deploy Qwen3-ASR to Google Cloud Run with a GPU. Windows/PowerShell twin of
    deploy.sh — same flags, same result.

.EXAMPLE
    .\deploy\gcp\deploy.ps1 -ProjectId my-project

.EXAMPLE
    # Redeploy an image that is already in Artifact Registry
    .\deploy\gcp\deploy.ps1 -ProjectId my-project -SkipBuild
#>
[CmdletBinding()]
param(
    [string]$ProjectId,
    [string]$Region = 'us-central1',
    [string]$Service = 'qwen3-asr',
    [string]$Repo = 'qwen3-asr',
    [string]$ImageName = 'qwen3-asr',
    [string]$Tag = 'v1',
    [string]$ServiceAccountName = 'qwen3-asr-sa',
    [string]$GpuType = 'nvidia-l4',
    [int]$Cpu = 8,
    [string]$Memory = '32Gi',
    [int]$MinInstances = 0,
    [int]$MaxInstances = 3,
    [int]$Concurrency = 1,
    [int]$Timeout = 3600,
    [switch]$AllowUnauthenticated,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'

if (-not $ProjectId) { $ProjectId = (gcloud config get-value project 2>$null) }
if (-not $ProjectId -or $ProjectId -eq '(unset)') {
    throw "No project set. Pass -ProjectId, or run: gcloud config set project YOUR_PROJECT"
}

$image = "$Region-docker.pkg.dev/$ProjectId/$Repo/${ImageName}:$Tag"
$serviceAccount = "$ServiceAccountName@$ProjectId.iam.gserviceaccount.com"

Write-Host "==> Project $ProjectId / region $Region / service $Service"
Write-Host "==> Image $image"

Write-Host '==> Enabling APIs'
gcloud services enable run.googleapis.com cloudbuild.googleapis.com `
    artifactregistry.googleapis.com storage.googleapis.com --project $ProjectId

Write-Host "==> Ensuring Artifact Registry repo '$Repo'"
gcloud artifacts repositories describe $Repo --location $Region --project $ProjectId 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    gcloud artifacts repositories create $Repo --repository-format=docker `
        --location $Region --description 'Qwen3-ASR Cloud Run images' --project $ProjectId
}

Write-Host "==> Ensuring runtime service account '$serviceAccount'"
gcloud iam service-accounts describe $serviceAccount --project $ProjectId 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) {
    gcloud iam service-accounts create $ServiceAccountName `
        --display-name 'Qwen3-ASR Cloud Run runtime' --project $ProjectId
}

# Needed only for gs:// audio inputs. Harmless otherwise, and scoped to read.
gcloud projects add-iam-policy-binding $ProjectId `
    --member "serviceAccount:$serviceAccount" `
    --role roles/storage.objectViewer --condition=None | Out-Null

if (-not $SkipBuild) {
    Write-Host '==> Building image (15-30 min; it downloads ~5 GB of weights)'
    gcloud builds submit --config deploy/gcp/cloudbuild.yaml --project $ProjectId `
        --substitutions "_REGION=$Region,_REPO=$Repo,_IMAGE=$ImageName,_TAG=$Tag" .
    if ($LASTEXITCODE -ne 0) { throw 'Cloud Build failed.' }
} else {
    Write-Host "==> -SkipBuild set, using existing $image"
}

$authFlag = '--no-allow-unauthenticated'
if ($AllowUnauthenticated) { $authFlag = '--allow-unauthenticated' }

Write-Host '==> Deploying to Cloud Run'
gcloud run deploy $Service `
    --image $image `
    --region $Region `
    --project $ProjectId `
    --platform managed `
    --execution-environment gen2 `
    --gpu 1 `
    --gpu-type $GpuType `
    --no-gpu-zonal-redundancy `
    --cpu $Cpu `
    --memory $Memory `
    --min-instances $MinInstances `
    --max-instances $MaxInstances `
    --concurrency $Concurrency `
    --timeout $Timeout `
    --port 8080 `
    --service-account $serviceAccount `
    --no-cpu-throttling `
    --set-env-vars "QWEN_PRELOAD_MODEL=true,QWEN_ENABLE_ALIGNER=true,QWEN_PRECISION=bfloat16,QWEN_MAX_CONCURRENCY=$Concurrency" `
    $authFlag
if ($LASTEXITCODE -ne 0) { throw 'Cloud Run deploy failed.' }

$url = gcloud run services describe $Service --region $Region --project $ProjectId --format 'value(status.url)'

Write-Host ''
Write-Host "==> Deployed: $url"
if ($AllowUnauthenticated) {
    Write-Host '    Service is PUBLIC. Set QWEN_AUTH_TOKEN to require a bearer token.'
} else {
    Write-Host '    Service requires IAM auth. Grant a caller:'
    Write-Host "      gcloud run services add-iam-policy-binding $Service --region $Region --member=user:you@example.com --role=roles/run.invoker"
}
Write-Host ''
Write-Host '    Smoke test:'
Write-Host "      `$t = gcloud auth print-identity-token"
Write-Host "      curl -H `"Authorization: Bearer `$t`" $url/ready"
