namespace OnnxSharp.Core;

/// <summary>运行时统一异常（对应 OpenCV 的 <c>CV_Error</c>）。</summary>
public class OnnxSharpException : Exception
{
    public OnnxSharpException(string message) : base(message)
    {
    }

    public OnnxSharpException(string message, Exception inner) : base(message, inner)
    {
    }
}
