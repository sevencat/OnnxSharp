<#
.SYNOPSIS
    Extract the PaddleOCR (PP-OCRv6) det / rec / dictionary (and optional cls) ONNX files
    from the nuget model packages, for the OnnxSharp `ocr` command and the OcrTests regression.

.DESCRIPTION
    The model packages (Sdcb.SimdPaddleOCR.Models.*) embed the ONNX files and the dictionary as
    assembly resources and contain **no inference code**, so the resources can be fed to any engine.
    This script copies them out unchanged:

        det.onnx    PP-OCRv6 tiny detector (DB)
        rec.onnx    PP-OCRv6 tiny recognizer (CTC, 6906 classes)
        dict.txt    PP-OCRv6 dictionary
        cls.onnx    PP-LCNet text-line orientation (from TextLineOrientation)

    Default destination is <repo>\.tmp\ocr-models (ignored by git). OcrTests auto-discovers that
    folder, or the one pointed to by the ONNXSHARP_OCR_MODELS environment variable.

.PARAMETER Destination
    Output directory. Default: <repo>\.tmp\ocr-models

.PARAMETER GlobalPackages
    nuget global packages folder. Default: whatever `dotnet nuget locals global-packages --list` says.

.EXAMPLE
    powershell -File tools\scripts\extract_ocr_models.ps1

.EXAMPLE
    powershell -File tools\scripts\extract_ocr_models.ps1 -Destination D:\models\ocr
#>
param(
    [string]$Destination,
    [string]$GlobalPackages
)

$ErrorActionPreference = 'Stop'

if (-not $Destination) {
    $repoRoot = Resolve-Path (Join-Path $PSScriptRoot '..\..')
    $Destination = Join-Path $repoRoot '.tmp\ocr-models'
}

if (-not $GlobalPackages) {
    $line = (& dotnet nuget locals global-packages --list | Out-String).Trim()
    $GlobalPackages = ($line -split ':', 2)[1].Trim()
}

if (-not (Test-Path -LiteralPath $GlobalPackages)) {
    throw "nuget global packages folder not found: $GlobalPackages (run 'dotnet restore' once)"
}

Write-Host "global packages : $GlobalPackages"
Write-Host "destination     : $Destination"

New-Item -ItemType Directory -Force -Path $Destination | Out-Null

function Export-ModelResources {
    param(
        [Parameter(Mandatory)] [string]$PackagePrefix,
        [Parameter(Mandatory)] [string]$AssemblyName
    )

    $dll = Get-ChildItem -LiteralPath $GlobalPackages -Recurse -Filter "$AssemblyName.dll" -ErrorAction SilentlyContinue |
        Where-Object { $_.FullName.ToLowerInvariant().Contains($PackagePrefix) } |
        Sort-Object FullName -Descending |
        Select-Object -First 1

    if (-not $dll) {
        Write-Warning "assembly $AssemblyName.dll not found (package $PackagePrefix not restored): skipped"
        return
    }

    $assembly = [System.Reflection.Assembly]::LoadFrom($dll.FullName)
    foreach ($name in $assembly.GetManifestResourceNames()) {
        # resource names look like Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny.det.onnx
        $fileName = ($name -split '\.')[-2..-1] -join '.'
        $target = Join-Path $Destination $fileName

        $stream = $assembly.GetManifestResourceStream($name)
        $file = [System.IO.File]::Create($target)
        try {
            $stream.CopyTo($file)
        }
        finally {
            $file.Dispose()
            $stream.Dispose()
        }

        $size = (Get-Item -LiteralPath $target).Length
        Write-Host ("  {0,-12} {1,12} B  <- {2}" -f $fileName, $size, $name)
    }
}

Export-ModelResources -PackagePrefix 'sdcb.simdpaddleocr.models.chinesev6tiny' -AssemblyName 'Sdcb.SimdPaddleOCR.Models.ChineseV6Tiny'
Export-ModelResources -PackagePrefix 'sdcb.simdpaddleocr.models.textlineorientation' -AssemblyName 'Sdcb.SimdPaddleOCR.Models.TextLineOrientation'

Write-Host ''
Write-Host 'usage:'
Write-Host "  OnnxSharp.Cli ocr page.bmp --det ...\det.onnx --rec ...\rec.onnx --dict ...\dict.txt"
