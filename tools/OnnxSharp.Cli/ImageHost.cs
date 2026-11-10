// CLI 的图像侧：一律走 ImageFacade（纯托管软后端 + Sdcb.SimdCodecs 编解码）。
// 只属于 CLI —— 产品库仍然「只吃像素」，不知道图片格式的存在。
using ImageFacade;
using ImageFacade.Backends;
using ImageFacade.SdcbSimdCodecs;
using Sdcb.SimdCodecs.Bmp;
using Sdcb.SimdCodecs.Jpeg;
using Sdcb.SimdCodecs.Png;

/// <summary>图像引擎与「一张已解码图片」的薄包装。</summary>
internal static class ImageHost
{
    private static ImageEngine? s_engine;
    private static bool s_codecs;

    /// <summary>
    /// 进程内共享一个纯托管图像引擎：软后端负责绘制 / 文字 / Resize，编解码按需注册。
    /// </summary>
    public static ImageEngine Engine
    {
        get
        {
            RegisterCodecs();
            return s_engine ??= new ImageEngine(new SoftwareBackend());
        }
    }

    /// <summary>
    /// 注册 CLI 需要的图片格式。格式包是**额外引入**的（<c>Sdcb.SimdCodecs.Jpeg / Png / Bmp</c>）：
    /// 软后端只带渲染与绘制，编解码必须显式注册，没注册的格式调用时抛 <c>FeatureNotSupportedException</c>。
    /// 注册是进程级静态且幂等，重复调用没有副作用；要支持更多格式就加包 + 在这里补一行。
    /// </summary>
    private static void RegisterCodecs()
    {
        if (s_codecs)
            return;
        s_codecs = true;

        ImageBackendBase.RegisterCodec(
            ImageFormat.Jpeg,
            JpegImageDecoder.Registration,
            JpegImageEncoder.Registration,
            request => new JpegEncodeOptions { Quality = Math.Clamp(request.Quality, 1, 100) });

        ImageBackendBase.RegisterCodec(
            ImageFormat.Png,
            PngImageDecoder.Registration,
            PngImageEncoder.Registration);

        ImageBackendBase.RegisterCodec(
            ImageFormat.Bmp,
            BmpImageDecoder.Registration,
            BmpImageEncoder.Registration);
    }
}

/// <summary>
/// 一张已解码的图 + 一份紧凑 RGB24 像素（引擎 / 检测器 / OCR 都吃这份缓冲）。
/// 原图保留在 <see cref="Image"/> 上，用来画框、缩放和保存。
/// </summary>
internal sealed class CliImage : IDisposable
{
    private CliImage(FacadeImage image)
    {
        Image = image;
        Pixels = image.CopyPixels(PixelFormat.Rgb24);
    }

    public static CliImage Load(string path) => new(ImageHost.Engine.Load(path));

    /// <summary>源图（可绘制、可缩放、可保存）。</summary>
    public FacadeImage Image { get; }

    /// <summary>紧凑 RGB24 像素，行跨度 = <c>Width * 3</c>。</summary>
    public byte[] Pixels { get; }

    public int Width => Image.Width;

    public int Height => Image.Height;

    public void Dispose() => Image.Dispose();
}
