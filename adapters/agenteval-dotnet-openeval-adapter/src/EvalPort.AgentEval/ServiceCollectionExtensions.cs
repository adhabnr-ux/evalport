// SPDX-License-Identifier: Apache-2.0

using AgentEval.Core;
using AgentEval.DataLoaders;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.DependencyInjection.Extensions;

namespace EvalPort.AgentEval;

/// <summary>
/// Registration of the EvalPort exporter and loader with AgentEval's DI.
/// </summary>
public static class ServiceCollectionExtensions
{
    /// <summary>
    /// Register <see cref="EvalPortResultExporter"/> as an <see cref="IResultExporter"/> and
    /// <see cref="EvalPortDatasetLoader"/> as an <see cref="IDatasetLoader"/>.
    /// </summary>
    /// <remarks>
    /// AgentEval picks these up in <c>services.AddAgentEvalDataLoaders()</c>, which builds the
    /// <c>IExporterRegistry</c> (every DI-registered <see cref="IResultExporter"/> is added under its
    /// <c>FormatName</c>, built-ins first) and the <c>IDatasetLoaderFactory</c> (every DI-registered
    /// loader under its extensions). Call order does not matter. In AgentEval 0.42.0-beta,
    /// <c>services.AddAgentEval()</c> alone does not create either registry, whatever its docs say;
    /// the test <c>DependencyInjectionTests</c> pins the behavior that is actually observed.
    /// </remarks>
    /// <param name="services">The service collection.</param>
    /// <param name="options">Export options; defaults to spec-conformant output.</param>
    public static IServiceCollection AddEvalPortAgentEval(this IServiceCollection services, EvalPortExportOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(services);
        var opts = options ?? EvalPortExportOptions.Default;
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IResultExporter, EvalPortResultExporter>(_ => new EvalPortResultExporter(opts)));
        services.TryAddEnumerable(ServiceDescriptor.Singleton<IDatasetLoader, EvalPortDatasetLoader>());
        return services;
    }
}
