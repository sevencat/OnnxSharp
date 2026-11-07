# 逐元素融合 A/B：同一会话内交替运行 base / fused 二进制，每个模型 3 轮。
#
# 两个二进制的准备方式（base = 融合前的提交，fused = 当前工作区）：
#   git worktree add --detach _base <融合前的 commit>
#   dotnet build _base/tools/OnnxSharp.Cli -c Release
#   dotnet build tools/OnnxSharp.Cli -c Release
#   mkdir bench/bin-base, bench/bin-fused
#   copy _base/tools/OnnxSharp.Cli/bin/Release/net10.0/* bench/bin-base/ -Recurse
#   copy tools/OnnxSharp.Cli/bin/Release/net10.0/* bench/bin-fused/ -Recurse
#   git worktree remove _base --force
# 必须交替跑（不是「跑完 base 再跑 fused」），否则跨时段的后台负载会污染结论。
$env:ONNXSHARP_THREADS = "16"

$models = @(
    @{ Name = "rfdetr-small";  Path = "F:/ai/rfdetr/onnx_model/rfdetr-small.onnx";  Input = "bench/inputs/i1x3x512x512.f32" },
    @{ Name = "rfdetr-medium"; Path = "F:/ai/rfdetr/onnx_model/rfdetr-medium.onnx"; Input = "bench/inputs/i1x3x576x576.f32" },
    @{ Name = "yolo11n";       Path = "F:/ai/models/yolo/yolo11n.onnx";              Input = "bench/inputs/i1x3x640x640.f32" },
    @{ Name = "yolov8s";       Path = "F:/ai/models/yolo/yolov8s.onnx";              Input = "bench/inputs/i1x3x640x640.f32" }
)

$rounds = 3
$iters = 20
$warmup = 5

function Parse-Bench([string]$text) {
    $min = [regex]::Match($text, 'min\s*:\s*([\d\.]+)\s*ms').Groups[1].Value
    $med = [regex]::Match($text, 'median\s*:\s*([\d\.]+)\s*ms').Groups[1].Value
    if ($min -eq '' -or $med -eq '') { throw "parse failed: $text" }
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
