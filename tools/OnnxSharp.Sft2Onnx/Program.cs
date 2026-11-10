using OnnxSharp.Core;
using OnnxSharp.Onnx;
using OnnxSharp.Safetensors;

namespace OnnxSharp.Sft2Onnx;

/// <summary>
/// safetensors → ONNX（权重补丁）命令行工具，纯 C#，不依赖 Python / onnx 包。
/// <para>
/// ONNX 文件 = 计算图 + 权重；safetensors 只有权重。所以「转换」= 拿一份<b>骨架</b> ONNX，
/// 把里面的 initializer 换成 safetensors 中的同名张量，图本身一字不改。
/// </para>
/// </summary>
internal static class Program
{
    private static int Main(string[] args)
    {
        try
        {
            if (args.Length == 0 || args[0] is "-h" or "--help" or "help")
                return Usage(args.Length == 0 ? 2 : 0);

            return args[0] switch
            {
                "patch" => Patch(args),
                "list" => List(args),
                _ => Usage(2),
            };
        }
        catch (OnnxSharpException e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return 1;
        }
        catch (IOException e)
        {
            Console.Error.WriteLine($"error: {e.Message}");
            return 1;
        }
    }

    private static int Usage(int code)
    {
        var writer = code == 0 ? Console.Out : Console.Error;
        writer.WriteLine("""
            用法:
              sft2onnx patch <skeleton.onnx> <weights.safetensors> -o <out.onnx> [选项]

                  --map <骨架名>=<权重名>   显式改名（可重复）
                  --strip a,b              匹配时去掉/补上的公共前缀（默认 module.,model.,_orig_mod.,backbone.）
                  --strict                 骨架里有、权重里没有的 initializer 直接报错
                  --external-threshold <G> 超过该体积（GiB，默认 1）改用外部数据文件
                  --data-name <file>       外部数据文件名（默认 <out.onnx>.data）
                  --verify                 写完用 OnnxSharp 加载一遍，确认结构没坏

              sft2onnx list <model.onnx | weights.safetensors>
                  按扩展名自动识别：ONNX 列出 initializer，safetensors 列出张量

            说明:
              ONNX = 计算图 + 权重，safetensors 只有权重。骨架提供「图」，本工具把「权重」换进去，
              图与其余字段（metadata / value_info / 未知扩展）按原始字节原样保留。

            典型用法（ultralytics 训练产物）:
              # 一次性（有 Python 的机器）：导出未折叠 BN 的骨架
              #   YOLO('best.pt').export(format='onnx', fuse=False, simplify=False)
              # 之后部署侧全 C#：
              sft2onnx patch best-fusefalse.onnx best.safetensors -o best.onnx --verify
            """);
        return code;
    }

    private static int Patch(string[] args)
    {
        if (args.Length < 3)
        {
            Console.Error.WriteLine("patch: needs <skeleton.onnx> and <weights.safetensors>");
            return Usage(2);
        }

        string skeletonPath = args[1];
        string weightsPath = args[2];

        string? output = null;
        string? dataName = null;
        bool strict = false;
        bool verify = false;
        double thresholdGiB = 1.0;
        var rename = new Dictionary<string, string>(StringComparer.Ordinal);
        var strips = new List<string>();

        for (int i = 3; i < args.Length; i++)
        {
            string arg = args[i];
            switch (arg)
            {
                case "-o":
                case "--output":
                    output = Next(args, ref i, arg);
                    break;

                case "--map":
                    {
                        string pair = Next(args, ref i, arg);
                        int split = pair.IndexOf('=');
                        if (split <= 0 || split == pair.Length - 1)
                        {
                            Console.Error.WriteLine($"--map expects <skeletonName>=<weightName>, got '{pair}'");
                            return 2;
                        }

                        rename[pair[..split]] = pair[(split + 1)..];
                        break;
                    }

                case "--strip":
                    strips.AddRange(Next(args, ref i, arg)
                        .Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries));
                    break;

                case "--strict":
                    strict = true;
                    break;

                case "--verify":
                    verify = true;
                    break;

                case "--external-threshold":
                    if (!double.TryParse(
                            Next(args, ref i, arg),
                            System.Globalization.NumberStyles.Float,
                            System.Globalization.CultureInfo.InvariantCulture,
                            out thresholdGiB))
                    {
                        Console.Error.WriteLine("--external-threshold expects a number (GiB)");
                        return 2;
                    }

                    break;

                case "--data-name":
                    dataName = Next(args, ref i, arg);
                    break;

                default:
                    Console.Error.WriteLine($"unknown option '{arg}'");
                    return Usage(2);
            }
        }

        if (output is null)
        {
            Console.Error.WriteLine("patch: missing -o <out.onnx>");
            return Usage(2);
        }

        var defaults = new OnnxWeightPatchOptions();
        var options = new OnnxWeightPatchOptions
        {
            StripPrefixes = strips.Count > 0 ? strips.ToArray() : defaults.StripPrefixes,
            ExternalDataThreshold = thresholdGiB <= 0 ? long.MaxValue : (long)(thresholdGiB * (1L << 30)),
            ExternalDataName = dataName,
            Strict = strict,
        };

        foreach (var (key, value) in rename)
            options.Rename[key] = value;

        return RunPatch(skeletonPath, weightsPath, output, options, verify);
    }

    private static int RunPatch(
        string skeletonPath, string weightsPath, string output, OnnxWeightPatchOptions options, bool verify)
    {
        var initializers = OnnxWeightPatcher.Inspect(skeletonPath);
        using var weights = SafeTensorFile.Open(weightsPath);

        var started = System.Diagnostics.Stopwatch.StartNew();
        var report = OnnxWeightPatcher.Apply(skeletonPath, weights, output, options);
        started.Stop();

        long weightBytes = 0;
        foreach (var tensor in weights.Tensors)
            weightBytes += tensor.ByteLength;

        Console.WriteLine($"skeleton  : {skeletonPath} ({initializers.Count} initializers)");
        Console.WriteLine($"weights   : {weightsPath} ({weights.Tensors.Count} tensors, {FormatBytes(weightBytes)})");
        Console.WriteLine($"output    : {output} ({new FileInfo(output).Length:N0} B)");
        Console.WriteLine(
            $"patched   : {report.Patched} / {initializers.Count} initializers, "
            + $"{(report.ExternalDataPath is null ? "inline" : "external")} "
            + FormatBytes(report.InlineBytes + report.ExternalBytes));

        if (report.ExternalDataPath is not null)
            Console.WriteLine($"data      : {report.ExternalDataPath} ({FormatBytes(report.ExternalBytes)})");

        Console.WriteLine(
            $"missing   : {(report.MissingInWeights.Count == 0 ? "(none)" : $"{report.MissingInWeights.Count} kept from skeleton")}");
        foreach (string name in report.MissingInWeights.Take(8))
            Console.WriteLine($"            - {name}");
        if (report.MissingInWeights.Count > 8)
            Console.WriteLine("            …");

        Console.WriteLine(
            $"unused    : {(report.UnusedWeights.Count == 0 ? "(none)" : report.UnusedWeights.Count.ToString())}");
        foreach (string name in report.UnusedWeights.Take(8))
            Console.WriteLine($"            - {name}");
        if (report.UnusedWeights.Count > 8)
            Console.WriteLine("            …");

        PrintHints(report);
        Console.WriteLine($"elapsed   : {started.ElapsedMilliseconds} ms");

        return verify ? Verify(output) : 0;
    }

    private static int List(string[] args)
    {
        if (args.Length < 2)
            return Usage(2);

        string path = args[1];

        if (path.EndsWith(".safetensors", StringComparison.OrdinalIgnoreCase))
        {
            using var file = SafeTensorFile.Open(path);
            long bytes = 0;
            foreach (var tensor in file.Tensors)
                bytes += tensor.ByteLength;

            Console.WriteLine($"{path}: {file.Tensors.Count} tensor(s), {FormatBytes(bytes)}");
            if (file.Metadata.Count > 0)
                Console.WriteLine($"metadata  : {string.Join(", ", file.Metadata.Select(m => $"{m.Key}={m.Value}"))}");

            foreach (var tensor in file.Tensors)
                Console.WriteLine($"  {tensor.Name,-58} {tensor.DType,-6} [{string.Join(",", tensor.Shape)}]");
            return 0;
        }

        var initializers = OnnxWeightPatcher.Inspect(path);
        Console.WriteLine($"{path}: {initializers.Count} initializer(s)");
        foreach (var tensor in initializers)
            Console.WriteLine($"  {tensor.Name,-58} type={tensor.OnnxDataType,-3} [{string.Join(",", tensor.Dims)}]");
        return 0;
    }

    private static int Verify(string path)
    {
        try
        {
            var net = Net.ReadModel(path);
            Console.WriteLine(
                $"verify    : OK — inputs [{string.Join(", ", net.InputNames)}], "
                + $"outputs [{string.Join(", ", net.OutputNames)}]");
            return 0;
        }
        catch (Exception e)
        {
            Console.Error.WriteLine($"verify    : FAILED — {e.GetType().Name}: {e.Message}");
            return 1;
        }
    }

    private static void PrintHints(OnnxWeightPatchReport report)
    {
        bool batchNormLeft = report.UnusedWeights.Any(
            n => n.Contains("running_mean", StringComparison.Ordinal)
                 || n.Contains("running_var", StringComparison.Ordinal));

        if (batchNormLeft)
        {
            Console.WriteLine();
            Console.WriteLine("hint      : 权重里有 BatchNorm 的 running_mean/running_var，但骨架里没有对应项。");
            Console.WriteLine("            骨架八成是用 fuse=True 导出的（BN 已被折进 Conv），这些权重会被忽略。");
            Console.WriteLine("            请用 fuse=False 重新导出骨架，例如 ultralytics:");
            Console.WriteLine("              YOLO('best.pt').export(format='onnx', fuse=False, simplify=False)");
        }

        if (report.MissingInWeights.Count > 0)
        {
            Console.WriteLine();
            Console.WriteLine("hint      : 未匹配的 initializer 保留骨架原值。若期望全部替换，检查命名（--map / --strip），");
            Console.WriteLine("            或加 --strict 让缺失直接报错。");
        }
    }

    private static string Next(string[] args, ref int index, string option)
        => index + 1 < args.Length
            ? args[++index]
            : throw new OnnxSharpException($"{option} needs a value");

    private static string FormatBytes(long bytes) => bytes switch
    {
        >= 1L << 30 => $"{bytes / (double)(1L << 30):F2} GiB",
        >= 1L << 20 => $"{bytes / (double)(1L << 20):F2} MiB",
        >= 1L << 10 => $"{bytes / (double)(1L << 10):F1} KiB",
        _ => $"{bytes} B",
    };
}
