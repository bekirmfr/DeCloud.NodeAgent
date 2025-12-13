using Microsoft.AspNetCore.Mvc;
using System.Diagnostics;
using System.Text;

namespace NodeAgent.Controllers;

/// <summary>
/// SSH Certificate Authority endpoints for the Node Agent
/// Signs SSH certificates using the node's SSH CA private key
/// </summary>
[ApiController]
[Route("api/ssh")]
public class SshCertificateController : ControllerBase
{
    private readonly ILogger<SshCertificateController> _logger;
    private const string CA_KEY_PATH = "/etc/ssh/decloud_ca";
    private const string CA_PUB_PATH = "/etc/ssh/decloud_ca.pub";

    public SshCertificateController(ILogger<SshCertificateController> logger)
    {
        _logger = logger;
    }

    /// <summary>
    /// Sign an SSH certificate using the node's CA
    /// </summary>
    [HttpPost("sign-certificate")]
    public async Task<ActionResult<CertificateSignResponse>> SignCertificate(
        [FromBody] CertificateSignRequest request)
    {
        try
        {
            _logger.LogInformation(
                "Signing SSH certificate {CertId} for principals: {Principals}",
                request.CertificateId,
                string.Join(", ", request.Principals));

            // Validate request
            if (string.IsNullOrEmpty(request.PublicKey))
            {
                return BadRequest(new CertificateSignResponse
                {
                    Success = false,
                    Error = "Public key is required"
                });
            }

            if (request.Principals == null || request.Principals.Count == 0)
            {
                return BadRequest(new CertificateSignResponse
                {
                    Success = false,
                    Error = "At least one principal is required"
                });
            }

            // Check if CA key exists
            if (!System.IO.File.Exists(CA_KEY_PATH))
            {
                _logger.LogError("SSH CA key not found at {Path}", CA_KEY_PATH);
                return StatusCode(500, new CertificateSignResponse
                {
                    Success = false,
                    Error = "SSH CA not configured on this node"
                });
            }

            // Create temporary file for public key
            var tempDir = Path.Combine(Path.GetTempPath(), "decloud-ssh");
            Directory.CreateDirectory(tempDir);

            var pubKeyFile = Path.Combine(tempDir, $"{request.CertificateId}.pub");
            var certFile = Path.Combine(tempDir, $"{request.CertificateId}-cert.pub");

            try
            {
                // Write public key to temp file
                await System.IO.File.WriteAllTextAsync(pubKeyFile, request.PublicKey);

                // Build ssh-keygen command
                var principals = string.Join(",", request.Principals);
                var validitySeconds = request.ValiditySeconds > 0 ? request.ValiditySeconds : 3600;

                // Build extensions string
                var extensions = new List<string>();
                foreach (var (key, value) in request.Extensions)
                {
                    if (string.IsNullOrEmpty(value))
                    {
                        extensions.Add(key);
                    }
                    else
                    {
                        extensions.Add($"{key}={value}");
                    }
                }
                var extensionsStr = string.Join(",", extensions);

                // ssh-keygen command to sign certificate
                var args = new List<string>
                {
                    "-s", CA_KEY_PATH,              // CA signing key
                    "-I", request.CertificateId,    // Certificate ID
                    "-n", principals,               // Principals
                    "-V", $"+{validitySeconds}s"    // Validity period
                };

                // Add extensions if any
                if (extensions.Count > 0)
                {
                    args.Add("-O");
                    args.Add(extensionsStr);
                }

                args.Add(pubKeyFile);               // Public key to sign

                var process = new Process
                {
                    StartInfo = new ProcessStartInfo
                    {
                        FileName = "ssh-keygen",
                        Arguments = string.Join(" ", args),
                        RedirectStandardOutput = true,
                        RedirectStandardError = true,
                        UseShellExecute = false,
                        CreateNoWindow = true
                    }
                };

                _logger.LogDebug("Running: ssh-keygen {Args}", string.Join(" ", args));

                process.Start();
                var stdout = await process.StandardOutput.ReadToEndAsync();
                var stderr = await process.StandardError.ReadToEndAsync();
                await process.WaitForExitAsync();

                if (process.ExitCode != 0)
                {
                    _logger.LogError(
                        "ssh-keygen failed with exit code {ExitCode}. Stderr: {Stderr}",
                        process.ExitCode,
                        stderr);

                    return StatusCode(500, new CertificateSignResponse
                    {
                        Success = false,
                        Error = $"Certificate signing failed: {stderr}"
                    });
                }

                // Read the signed certificate
                if (!System.IO.File.Exists(certFile))
                {
                    _logger.LogError("Certificate file not created: {Path}", certFile);
                    return StatusCode(500, new CertificateSignResponse
                    {
                        Success = false,
                        Error = "Certificate file not created"
                    });
                }

                var signedCertificate = await System.IO.File.ReadAllTextAsync(certFile);

                var validUntil = DateTime.UtcNow.AddSeconds(validitySeconds);

                _logger.LogInformation(
                    "✓ Certificate {CertId} signed successfully, valid until {ValidUntil}",
                    request.CertificateId,
                    validUntil);

                return Ok(new CertificateSignResponse
                {
                    Success = true,
                    SignedCertificate = signedCertificate.Trim(),
                    ValidUntil = validUntil
                });
            }
            finally
            {
                // Clean up temporary files
                try
                {
                    if (System.IO.File.Exists(pubKeyFile))
                        System.IO.File.Delete(pubKeyFile);
                    if (System.IO.File.Exists(certFile))
                        System.IO.File.Delete(certFile);
                }
                catch (Exception ex)
                {
                    _logger.LogWarning(ex, "Failed to clean up temporary files");
                }
            }
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error signing SSH certificate");
            return StatusCode(500, new CertificateSignResponse
            {
                Success = false,
                Error = $"Internal error: {ex.Message}"
            });
        }
    }

    /// <summary>
    /// Get CA public key for client verification
    /// </summary>
    [HttpGet("ca-public-key")]
    public async Task<ActionResult<CaPublicKeyResponse>> GetCaPublicKey()
    {
        try
        {
            if (!System.IO.File.Exists(CA_PUB_PATH))
            {
                return NotFound(new CaPublicKeyResponse
                {
                    Success = false,
                    Error = "CA public key not found"
                });
            }

            var publicKey = await System.IO.File.ReadAllTextAsync(CA_PUB_PATH);

            return Ok(new CaPublicKeyResponse
            {
                Success = true,
                PublicKey = publicKey.Trim()
            });
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error reading CA public key");
            return StatusCode(500, new CaPublicKeyResponse
            {
                Success = false,
                Error = $"Failed to read CA public key: {ex.Message}"
            });
        }
    }
}

#region DTOs

public class CertificateSignRequest
{
    public string PublicKey { get; set; } = "";
    public string CertificateId { get; set; } = "";
    public List<string> Principals { get; set; } = new();
    public int ValiditySeconds { get; set; } = 3600;
    public Dictionary<string, string> Extensions { get; set; } = new();
}

public class CertificateSignResponse
{
    public bool Success { get; set; }
    public string SignedCertificate { get; set; } = "";
    public string? Error { get; set; }
    public DateTime? ValidUntil { get; set; }
}

public class CaPublicKeyResponse
{
    public bool Success { get; set; }
    public string PublicKey { get; set; } = "";
    public string? Error { get; set; }
}

#endregion