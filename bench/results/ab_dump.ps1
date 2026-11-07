# base / fused 逐值比对：每个模型各跑一次并把全部输出落盘，再交给 cmp_all.py 比对。
$env:ONNXSHARP_THREADS = "16"

$models = @(
    @{ Name = "rfdetr-small";  Path = "F:/ai/rfdetr/onnx_model/rfdetr-small.onnx";  Input = "bench/inputs/i1x3x512x512.f32" },
    @{ Name = "rfdetr-medium"; Path = "F:/ai/rfdetr/onnx_model/rfdetr-medium.onnx"; Input = "bench/inputs/i1x3x576x576.f32" },
    @{ Name = "yolo11n";       Path = "F:/ai/models/yolo/yolo11n.onnx";              Input = "bench/inputs/i1x3x640x640.f32" },
    @{ Name = "yolov8s";       Path = "F:/ai/models/yolo/yolov8s.onnx";              Input = "bench/inputs/i1x3x640x640.f32" }
)

foreach ($m in $models) {
    foreach ($side in @("base", "fused")) {
        $dir = "bench/results/dump-$($m.Name)-$side"
        & "bench/bin-$side/OnnxSharp.Cli.exe" run $m.Path --input $m.Input --dump $dir 2>&1 | Out-Null
    }
    "dumped $($m.Name)"
}
