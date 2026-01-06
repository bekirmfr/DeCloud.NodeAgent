using Microsoft.Extensions.Configuration;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using Nethereum.Signer;
using Nethereum.Util;
using System.Security.Cryptography;
using System.Text;

namespace DeCloud.NodeAgent.Infrastructure.Services.Auth;

/// <summary>
/// Service interface for node wallet operations
/// </summary>
public interface INodeWalletService
{
    /// <summary>
    /// Get the node's wallet address
    /// </summary>
    string GetWalletAddress();

    /// <summary>
    /// Sign a message with the node's private key
    /// </summary>
    Task<string> SignMessageAsync(string message);

    /// <summary>
    /// Verify that a signature is valid
    /// </summary>
    bool VerifySignature(string message, string signature, string expectedAddress);
}

/// <summary>
/// Manages node wallet operations for authentication.
/// Loads private key from secure storage and signs messages.
/// </summary>
public class NodeWalletService : INodeWalletService
{
    private readonly EthECKey _privateKey;
    private readonly string _walletAddress;
    private readonly ILogger<NodeWalletService> _logger;

    public NodeWalletService(
        IConfiguration configuration,
        ILogger<NodeWalletService> logger)
    {
        _logger = logger;

        // Load private key from configuration
        var privateKeyHex = configuration["Node:PrivateKey"];

        if (string.IsNullOrEmpty(privateKeyHex))
        {
            throw new InvalidOperationException(
                "Node:PrivateKey not configured. " +
                "Set environment variable NODE_PRIVATE_KEY or add to appsettings.json");
        }

        // Validate and normalize private key format
        if (!privateKeyHex.StartsWith("0x"))
        {
            privateKeyHex = "0x" + privateKeyHex;
        }

        if (privateKeyHex.Length != 66) // 0x + 64 hex chars
        {
            throw new InvalidOperationException(
                "Invalid private key format. Must be 64 hex characters (optionally prefixed with 0x)");
        }

        try
        {
            // Create EthECKey from private key
            _privateKey = new EthECKey(privateKeyHex);

            // Get the public address
            _walletAddress = _privateKey.GetPublicAddress();

            _logger.LogInformation(
                "✓ Node wallet initialized: {Address}",
                _walletAddress);
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException(
                "Failed to initialize wallet from private key. " +
                "Ensure NODE_PRIVATE_KEY is a valid Ethereum private key.", ex);
        }
    }

    public string GetWalletAddress()
    {
        return _walletAddress;
    }

    /// <summary>
    /// Sign a message using Ethereum personal_sign (EIP-191)
    /// </summary>
    public async Task<string> SignMessageAsync(string message)
    {
        try
        {
            var signer = new EthereumMessageSigner();
            var signature = signer.EncodeUTF8AndSign(message, _privateKey);

            _logger.LogDebug("Signed message: {MessagePreview}...",
                message.Substring(0, Math.Min(50, message.Length)));

            return await Task.FromResult(signature);
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Failed to sign message");
            throw new InvalidOperationException("Message signing failed", ex);
        }
    }

    /// <summary>
    /// Verify a signature is valid for a message and address
    /// </summary>
    public bool VerifySignature(string message, string signature, string expectedAddress)
    {
        try
        {
            var signer = new EthereumMessageSigner();
            var recoveredAddress = signer.EncodeUTF8AndEcRecover(message, signature);

            var isValid = string.Equals(
                recoveredAddress,
                expectedAddress,
                StringComparison.OrdinalIgnoreCase);

            if (!isValid)
            {
                _logger.LogWarning(
                    "Signature verification failed. Expected: {Expected}, Recovered: {Recovered}",
                    expectedAddress, recoveredAddress);
            }

            return isValid;
        }
        catch (Exception ex)
        {
            _logger.LogError(ex, "Error verifying signature");
            return false;
        }
    }
}

/// <summary>
/// Extension methods for wallet configuration
/// </summary>
public static class NodeWalletServiceExtensions
{
    /// <summary>
    /// Add NodeWalletService to dependency injection
    /// </summary>
    public static IServiceCollection AddNodeWallet(this IServiceCollection services)
    {
        services.AddSingleton<INodeWalletService, NodeWalletService>();
        return services;
    }
}