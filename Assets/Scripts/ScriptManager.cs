using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class ScriptManager : MonoBehaviour
{
    private AgentSpawner agentSpawner;
    private Simulator simulator;
    
    void Start()
    {
        agentSpawner = GetComponent<AgentSpawner>();
        simulator = GetComponent<Simulator>();


        List<Agent> agents = agentSpawner.SpawnAgents();
        simulator.RegisterAgents(agents);
        simulator.StartSimulation();
    }
}
