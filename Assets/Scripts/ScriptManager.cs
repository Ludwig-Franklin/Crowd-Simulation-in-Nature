using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScriptManager : MonoBehaviour
{
    private AgentSpawner agentSpawner;
    private Simulator simulator;
    private ExperimentManager experimentManager;
    
    void Start()
    {
        agentSpawner = GetComponent<AgentSpawner>();
        simulator = GetComponent<Simulator>();
        experimentManager = GetComponent<ExperimentManager>();

        // Only start simulation automatically if we're not running an experiment
        if (experimentManager == null || !experimentManager.runExperiment)
        {
            StartSimulation();
        }
    }
    
    public void StartSimulation()
    {
        List<Agent> agents = agentSpawner.SpawnAgents();
        simulator.RegisterAgents(agents);
        simulator.StartSimulation();
    }

    public void OnAgentSpawned(Agent agent)
    {
        // Register the agent with the simulator
        if (simulator != null && agent != null)
        {
            simulator.RegisterSingleAgent(agent);
        }
    }
}
