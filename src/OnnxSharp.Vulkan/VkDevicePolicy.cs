using OnnxSharp.Core;

namespace OnnxSharp.Vulkan;

/// <summary>
/// 设备门。与 <see cref="VkOpRegistry.CanEmit"/> 分工明确：
/// CanEmit 回答「**能不能**发射」（dtype / 形状 / 属性支持不支持），
/// 这里回答「**值不值得**」（这个设备上跑这层是赚还是亏）。
/// <para>
/// 目前只有一条规则，来自 benchv3.md 的实测而不是猜测：核显上 Conv 3×3 慢 3–20 倍，
/// 1×1 特化（P2-2）之后仍慢 5–7 倍；而 CPU 侧的分块卷积把 1×1 直接当 GEMM 走，
/// 这个量级（几十 MFLOP）很难被朴素核追上。所以在 P2-3（NHWC + 向量载入 + shared memory 分块）
/// 落地之前，核显一律不发 Conv——宁可整段回落 CPU，也不要在核显上排一串必输的 dispatch。
/// </para>
/// <para>
/// 逐元素算子**不受此限**：它们是段的主体，且在核显上仍快于 CPU（见探针 eltwise cost）。
/// 这一条很重要——设备门不能一刀切成「核显不用 GPU」，否则会连本来赚钱的段一起丢掉。
/// </para>
/// <para>
/// 已知边界：<see cref="VkDevice.UnifiedMemory"/> 只认 <c>INTEGRATED</c>，<c>OTHER</c>/<c>CPU</c>
/// 类型的设备（lavapipe 之类）会被当成独显放行。这类设备本来就慢，真要支持得按
/// 「是否软件实现」单独判，暂不在本轮范围内。
/// </para>
/// </summary>
internal static class VkDevicePolicy
{
    /// <summary>
    /// 判断 <paramref name="op"/> 是否允许在该设备上发射。<paramref name="reason"/> 在拒绝时给出原因，
    /// 允许时为空串（探针把它打出来，免得设备门变成「悄悄不发射」的黑盒）。
    /// </summary>
    public static bool Allows(VkDevice dev, LayerInfo op, out string reason)
    {
        ArgumentNullException.ThrowIfNull(dev);
        ArgumentNullException.ThrowIfNull(op);

        reason = "";
        if (!dev.UnifiedMemory) return true;

        switch (op.Type)
        {
            case "Conv":
                reason = $"核显（{dev.DeviceName}）上 Conv 实测劣于 CPU，暂不发 GPU";
                return false;

            default:
                return true;
        }
    }
}
