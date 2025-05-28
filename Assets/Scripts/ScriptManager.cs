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
        // Let the AgentSpawner handle spawning and registering agents
        List<Agent> agents = agentSpawner.SpawnAgents();
        
        // Just start the simulation - the agents are already registered by the AgentSpawner
        simulator.StartSimulation();
    }
}
