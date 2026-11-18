namespace OnnxSharp.Vulkan;

/// <summary>
/// 设备丢失（驱动重置 / GPU 故障）。该设备上创建的所有对象都已失效，
/// 调用方应丢弃 <c>VkPartitionLayer</c> 并回退到 CPU 执行。
/// </summary>
public sealed class VkDeviceLostException(string what)
    : InvalidOperationException($"Vulkan {what} failed: ErrorDeviceLost");
