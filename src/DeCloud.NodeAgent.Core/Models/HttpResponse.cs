using Orchestrator.Models;
using System.Data;
using System.Reflection.Metadata;
using System.Text.Json;

namespace DeCloud.NodeAgent.Core.Models
{
    public class HttpResponse<T> where T : class
    {
        public bool IsSuccess { get; init; }
        public T? Data { get; init; }
        public string? Error { get; init; }

        public static HttpResponse<T> Success(T data) =>
            new() { IsSuccess = true, Data = data };

        public static HttpResponse<T> Failure(string error) =>
            new() { IsSuccess = false, Error = error };

        public async static Task<HttpResponse<T>> FromResponseAsync(HttpResponseMessage response)
        {
            try
            {
                if (!response.IsSuccessStatusCode)
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    return Failure($"Http request failed with status code {response.StatusCode}: {errorContent}");
                }

                var content = await response.Content.ReadAsStringAsync();
                var json = JsonDocument.Parse(content);

                if (!json.RootElement.TryGetProperty("data", out var dataJson))
                {
                    throw new FormatException("Http response missing 'data' property");
                }

                var data = JsonSerializer.Deserialize<T>(
                        dataJson.GetRawText(),
                        new JsonSerializerOptions
                        {
                            PropertyNameCaseInsensitive = true
                        });
                if (data == null)
                {
                    throw new ArgumentException("Http response 'data' property could not be deserialized");
                }

                return Success(data);
            }
            catch (Exception ex)
            {
                return Failure($"Http response could not be processed: {ex.Message}");
            }

        }

        public static HttpResponse<T> FromException(Exception ex) =>
            Failure($"Http request failed: {ex.Message}");
    }
}
