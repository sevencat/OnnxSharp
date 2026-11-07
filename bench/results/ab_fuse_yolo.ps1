# yolo 系列 A/B（长跑）：确认逐元素融合不引入回归。5 轮 × 50 iter 交替。
$env:ONNXSHARP_THREADS = "16"

$models = @(
    @{ Name = "yolo11n"; Path = "F:/ai/models/yolo/yolo11n.onnx"; Input = "bench/inputs/i1x3x640x640.f32" },
    @{ Name = "yolov8s"; Path = "F:/ai/models/yolo/yolov8s.onnx"; Input = "bench/inputs/i1x3x640x640.f32" }
)

$rounds = 5
$iters = 50
$warmup = 10

function Parse-Bench([string]$text) {
    $min = [regex]::Match($text, 'min\s*:\s*([\d\.]+)\s*ms').Groups[1].Value
    $med = [regex]::Match($text, 'median\s*:\s*([\d\.]+)\s*ms').Groups[1].Value
    if ($min -eq '' -or $med -eq '') { throw "parse failed" }
    return [pscustomobject]@{ min = [double]$min; median = [double]$med }
}

function Get-Median($values) {
    $s = @($values | Sort-Object)
    return $s[[int][math]::Floor($s.Count / 2)]
}

$rows = @()

foreach ($m in $models) {
    $res = @{ base = @(); fused = @() }
    for ($r = 0; $r -lt $rounds; $r++) {
        foreach ($side in @("base", "fused")) {
            $text = (& "bench/bin-$side/OnnxSharp.Cli.exe" bench $m.Path --input $m.Input --iters $iters --warmup $warmup 2>&1) -join "`n"
            $res[$side] += Parse-Bench $text
        }
    }

    $bMin = Get-Median ($res.base | ForEach-Object { $_.min })
    $fMin = Get-Median ($res.fused | ForEach-Object { $_.min })
    $bMed = Get-Median ($res.base | ForEach-Object { $_.median })
    $fMed = Get-Median ($res.fused | ForEach-Object { $_.median })

    $rows += [pscustomobject]@{
        model     = $m.Name
        base_min  = $bMin
        fused_min = $fMin
        base_med  = $bMed
        fused_med = $fMed
        min_gain  = "{0:+0.0;-0.0}%" -f (($bMin - $fMin) / $bMin * 100)
        med_gain  = "{0:+0.0;-0.0}%" -f (($bMed - $fMed) / $bMed * 100)
    }
}

$rows | Format-Table -AutoSize | Out-String -Width 200
