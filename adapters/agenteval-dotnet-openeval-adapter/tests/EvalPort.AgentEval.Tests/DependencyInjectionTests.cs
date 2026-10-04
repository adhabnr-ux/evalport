// SPDX-License-Identifier: Apache-2.0

using AgentEval.Core;
using AgentEval.DataLoaders;
using AgentEval.DependencyInjection;
using Microsoft.Extensions.DependencyInjection;

namespace EvalPort.AgentEval.Tests;

/// <summary>How the exporter and loader reach AgentEval's registries through DI, in AgentEval 0.42.0-beta.</summary>
public class DependencyInjectionTests
{
    [Fact]
    public void AddAgentEvalDataLoaders_PicksUpTheExporterAndTheLoader_InEitherOrder()
    {
        foreach (var ourFirst in new[] { true, false })
        {
            var services = new ServiceCollection();
            if (ourFirst) services.AddEvalPortAgentEval();
            services.AddAgentEvalDataLoaders();
            if (!ourFirst) services.AddEvalPortAgentEval();
            using var provider = services.BuildServiceProvider();

            var registry = provider.GetRequiredService<IExporterRegistry>();
            Assert.True(registry.Contains("evalport"));
            Assert.IsType<EvalPortResultExporter>(registry.GetRequired("evalport"));
            // Built-ins are untouched.
            Assert.True(registry.Contains("Json"));

            var factory = provider.GetRequiredService<IDatasetLoaderFactory>();
            Assert.IsType<EvalPortDatasetLoader>(factory.CreateFromExtension(".evalport.json"));
        }
    }

    [Fact]
    public void AddEvalPortAgentEval_Twice_RegistersOnce()
    {
        var services = new ServiceCollection().AddEvalPortAgentEval().AddEvalPortAgentEval();
        Assert.Single(services, d => d.ServiceType == typeof(IResultExporter));
        Assert.Single(services, d => d.ServiceType == typeof(IDatasetLoader));
    }

    [Fact]
    public void Options_FlowThroughDi()
    {
        var services = new ServiceCollection().AddEvalPortAgentEval(new EvalPortExportOptions { EmitProposedVerdict = true });
        using var provider = services.BuildServiceProvider();
        var exporter = Assert.IsType<EvalPortResultExporter>(provider.GetRequiredService<IResultExporter>());
        var rs = exporter.ToResultSet(new global::AgentEval.Models.EvaluationReport
        {
            TestResults = { new global::AgentEval.Models.TestResultSummary { Name = "t", Passed = true, Score = 100 } },
        });
        Assert.Equal("passed", (string)rs["results"]![0]!["verdict"]!);
    }

    [Fact]
    public void AddAgentEvalAlone_DoesNotCreateTheRegistries_WhateverTheDocsSay()
    {
        // docs/export.md and docs/extensibility.md in AgentEval say AddAgentEval() populates IExporterRegistry;
        // in 0.42.0-beta only AddAgentEvalDataLoaders() does. Pinned here so the README's advice stays true.
        var services = new ServiceCollection().AddEvalPortAgentEval();
        services.AddAgentEval();
        using var provider = services.BuildServiceProvider();
        Assert.Null(provider.GetService<IExporterRegistry>());
        Assert.Null(provider.GetService<IDatasetLoaderFactory>());
    }
}
