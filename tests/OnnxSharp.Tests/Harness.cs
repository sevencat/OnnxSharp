namespace OnnxSharp.Tests;

/// <summary>一个测试用例：名字 + 主体（抛异常即失败）。</summary>
internal readonly record struct TestCase(string Name, Action Body);

/// <summary>
/// 极简测试运行器：不依赖 xunit/NUnit，离线可跑、AOT 友好（无反射）。
/// <c>dotnet run --project tests/OnnxSharp.Tests</c>，退出码 0 = 全通过。
/// </summary>
internal static class Harness
{
    public static int RunAll(IEnumerable<TestCase> cases)
    {
        var list = cases.ToList();
        int passed = 0;
        var failures = new List<string>();

        Console.WriteLine($"running {list.Count} test(s)");
        foreach (var test in list)
        {
            try
            {
                test.Body();
                passed++;
                Console.WriteLine($"  PASS  {test.Name}");
            }
            catch (Exception e)
            {
                failures.Add(test.Name);
                Console.WriteLine($"  FAIL  {test.Name}");
                Console.WriteLine($"        {e.GetType().Name}: {e.Message}");
            }
        }

        Console.WriteLine();
        Console.WriteLine($"{passed}/{list.Count} passed, {failures.Count} failed");
        if (failures.Count > 0)
        {
            Console.WriteLine("failed:");
            foreach (var name in failures)
                Console.WriteLine($"  - {name}");
        }

        return failures.Count == 0 ? 0 : 1;
    }
}

internal static class Assert
{
    public static void True(bool condition, string message)
    {
        if (!condition)
            throw new TestFailureException(message);
    }

    public static void False(bool condition, string message) => True(!condition, message);

    public static void Equal<T>(T expected, T actual, string message)
    {
        if (!EqualityComparer<T>.Default.Equals(expected, actual))
            throw new TestFailureException($"{message}: expected {expected}, got {actual}");
    }

    public static void NotEqual<T>(T unexpected, T actual, string message)
    {
        if (EqualityComparer<T>.Default.Equals(unexpected, actual))
            throw new TestFailureException($"{message}: both are {actual}");
    }

    public static void Near(float expected, float actual, float tolerance, string message)
    {
        if (MathF.Abs(expected - actual) > tolerance)
            throw new TestFailureException($"{message}: expected {expected} ± {tolerance}, got {actual}");
    }

    public static void Throws<TException>(Action action, string message) where TException : Exception
    {
        try
        {
            action();
        }
        catch (TException)
        {
            return;
        }
        catch (Exception e)
        {
            throw new TestFailureException($"{message}: expected {typeof(TException).Name}, got {e.GetType().Name}");
        }

        throw new TestFailureException($"{message}: expected {typeof(TException).Name}, nothing thrown");
    }

    public static void Same(object? expected, object? actual, string message)
    {
        if (!ReferenceEquals(expected, actual))
            throw new TestFailureException($"{message}: references differ");
    }

    public static void NotSame(object? expected, object? actual, string message)
    {
        if (ReferenceEquals(expected, actual))
            throw new TestFailureException($"{message}: references are the same");
    }
}

internal sealed class TestFailureException(string message) : Exception(message);
